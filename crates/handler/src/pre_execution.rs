//! Handles related to the main function of the EVM.
//!
//! They handle initial setup of the EVM, call loop and the final return of the EVM

use bytecode::Bytecode;
use context_interface::{
    journaled_state::Journal,
    result::InvalidTransaction,
    transaction::{Transaction, TransactionType},
    Block, BlockGetter, Cfg, CfgGetter, Database, DatabaseGetter, JournalDBError, JournalGetter,
    PerformantContextAccess, TransactionGetter,
};
use handler_interface::PreExecutionHandler;
use primitives::{Address, BLOCKHASH_STORAGE_ADDRESS, KECCAK_EMPTY, U256};
use specification::{eip7702, hardfork::SpecId};
use std::{boxed::Box, vec::Vec};

#[derive(Default)]
pub struct EthPreExecution<CTX, ERROR> {
    pub _phantom: core::marker::PhantomData<(CTX, ERROR)>,
}

impl<CTX, ERROR> EthPreExecution<CTX, ERROR> {
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }

    pub fn new_boxed() -> Box<Self> {
        Box::new(Self::new())
    }
}

impl<CTX, ERROR> PreExecutionHandler for EthPreExecution<CTX, ERROR>
where
    CTX: EthPreExecutionContext,
    ERROR: EthPreExecutionError<CTX>,
{
    type Context = CTX;
    type Error = ERROR;

    fn load_accounts(&self, context: &mut Self::Context) -> Result<(), Self::Error> {
        let spec = context.cfg().spec().into();
        // Set journaling state flag.
        context.journal().set_spec_id(spec);

        // Load coinbase
        // EIP-3651: Warm COINBASE. Starts the `COINBASE` address warm
        if spec.is_enabled_in(SpecId::SHANGHAI) {
            let coinbase = context.block().beneficiary();
            context.journal().warm_account(coinbase);
        }

        // Load blockhash storage address
        // EIP-2935: Serve historical block hashes from state
        if spec.is_enabled_in(SpecId::PRAGUE) {
            context.journal().warm_account(BLOCKHASH_STORAGE_ADDRESS);
        }

        // Load access list
        context.load_access_list()?;

        Ok(())
    }

    fn apply_eip7702_auth_list(&self, context: &mut Self::Context) -> Result<u64, Self::Error> {
        let spec = context.cfg().spec().into();
        if spec.is_enabled_in(SpecId::PRAGUE) {
            apply_eip7702_auth_list::<CTX, ERROR>(context)
        } else {
            Ok(0)
        }
    }

    #[inline]
    fn deduct_caller(&self, context: &mut Self::Context) -> Result<(), Self::Error> {
        let basefee = context.block().basefee();
        let blob_price = context.block().blob_gasprice().unwrap_or_default();
        let effective_gas_price = context.tx().effective_gas_price(basefee as u128);
        // Subtract gas costs from the caller's account.
        // We need to saturate the gas cost to prevent underflow in case that `disable_balance_check` is enabled.
        let mut gas_cost = (context.tx().gas_limit() as u128).saturating_mul(effective_gas_price);

        // EIP-4844
        if context.tx().tx_type() == TransactionType::Eip4844 {
            let blob_gas = context.tx().total_blob_gas() as u128;
            gas_cost = gas_cost.saturating_add(blob_price.saturating_mul(blob_gas));
        }

        let is_call = context.tx().kind().is_call();
        let caller = context.tx().caller();

        // Load caller's account.
        let caller_account = context.journal().load_account(caller)?.data;
        // Set new caller account balance.
        caller_account.info.balance = caller_account
            .info
            .balance
            .saturating_sub(U256::from(gas_cost));

        // Bump the nonce for calls. Nonce for CREATE will be bumped in `handle_create`.
        if is_call {
            // Nonce is already checked
            caller_account.info.nonce = caller_account.info.nonce.saturating_add(1);
        }

        // Touch account so we know it is changed.
        caller_account.mark_touch();
        Ok(())
    }
}

/// Apply EIP-7702 auth list and return number gas refund on already created accounts.
#[inline]
pub fn apply_eip7702_auth_list<
    CTX: TransactionGetter + JournalGetter + CfgGetter,
    ERROR: From<InvalidTransaction> + From<JournalDBError<CTX>>,
>(
    context: &mut CTX,
) -> Result<u64, ERROR> {
    // Return if there is no auth list.
    let tx = context.tx();
    if tx.tx_type() != TransactionType::Eip7702 {
        return Ok(0);
    }

    struct Authorization {
        authority: Option<Address>,
        address: Address,
        nonce: u64,
        chain_id: U256,
    }

    let authorization_list = tx
        .authorization_list()
        .map(|a| Authorization {
            authority: a.0,
            chain_id: a.1,
            nonce: a.2,
            address: a.3,
        })
        .collect::<Vec<_>>();
    let chain_id = context.cfg().chain_id();

    let mut refunded_accounts = 0;
    for authorization in authorization_list {
        // 1. Verify the chain id is either 0 or the chain's current ID.
        let auth_chain_id = authorization.chain_id;
        if !auth_chain_id.is_zero() && auth_chain_id != U256::from(chain_id) {
            continue;
        }

        // 2. Verify the `nonce` is less than `2**64 - 1`.
        if authorization.nonce == u64::MAX {
            continue;
        }

        // recover authority and authorized addresses.
        // 3. `authority = ecrecover(keccak(MAGIC || rlp([chain_id, address, nonce])), y_parity, r, s]`
        let Some(authority) = authorization.authority else {
            continue;
        };

        // warm authority account and check nonce.
        // 4. Add `authority` to `accessed_addresses` (as defined in [EIP-2929](./eip-2929.md).)
        let mut authority_acc = context.journal().load_account_code(authority)?;

        // 5. Verify the code of `authority` is either empty or already delegated.
        if let Some(bytecode) = &authority_acc.info.code {
            // if it is not empty and it is not eip7702
            if !bytecode.is_empty() && !bytecode.is_eip7702() {
                continue;
            }
        }

        // 6. Verify the nonce of `authority` is equal to `nonce`. In case `authority` does not exist in the trie, verify that `nonce` is equal to `0`.
        if authorization.nonce != authority_acc.info.nonce {
            continue;
        }

        // 7. Add `PER_EMPTY_ACCOUNT_COST - PER_AUTH_BASE_COST` gas to the global refund counter if `authority` exists in the trie.
        if !authority_acc.is_empty() {
            refunded_accounts += 1;
        }

        // 8. Set the code of `authority` to be `0xef0100 || address`. This is a delegation designation.
        //  * As a special case, if `address` is `0x0000000000000000000000000000000000000000` do not write the designation. Clear the accounts code and reset the account's code hash to the empty hash `0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470`.
        let (bytecode, hash) = if authorization.address.is_zero() {
            (Bytecode::default(), KECCAK_EMPTY)
        } else {
            let bytecode = Bytecode::new_eip7702(authorization.address);
            let hash = bytecode.hash_slow();
            (bytecode, hash)
        };
        authority_acc.info.code_hash = hash;
        authority_acc.info.code = Some(bytecode);

        // 9. Increase the nonce of `authority` by one.
        authority_acc.info.nonce = authority_acc.info.nonce.saturating_add(1);
        authority_acc.mark_touch();
    }

    let refunded_gas =
        refunded_accounts * (eip7702::PER_EMPTY_ACCOUNT_COST - eip7702::PER_AUTH_BASE_COST);

    Ok(refunded_gas)
}

pub trait EthPreExecutionContext:
    TransactionGetter
    + BlockGetter
    + JournalGetter
    + CfgGetter
    + PerformantContextAccess<Error = <<Self as DatabaseGetter>::Database as Database>::Error>
{
}

impl<
        CTX: TransactionGetter
            + BlockGetter
            + JournalGetter
            + CfgGetter
            + PerformantContextAccess<Error = <<CTX as DatabaseGetter>::Database as Database>::Error>,
    > EthPreExecutionContext for CTX
{
}

pub trait EthPreExecutionError<CTX: JournalGetter>:
    From<InvalidTransaction> + From<JournalDBError<CTX>>
{
}

impl<CTX: JournalGetter, T: From<InvalidTransaction> + From<JournalDBError<CTX>>>
    EthPreExecutionError<CTX> for T
{
}
