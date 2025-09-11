// src/handlers/slack/encryption.rs

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ring::{
    aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM},
    rand::{SecureRandom, SystemRandom},
};
use std::env;
// use tracing::error;

use crate::handlers::slack::ENCRYPTION_KEY;

pub fn initialize_encryption_key() -> Result<(), Box<dyn std::error::Error>> {
    let key_material = if let Ok(kms_key_id) = env::var("KMS_KEY_ID") {
        fetch_key_from_kms(&kms_key_id)?
    } else if let Ok(vault_path) = env::var("VAULT_KEY_PATH") {
        fetch_key_from_vault(&vault_path)?
    } else if let Ok(key_base64) = env::var("ENCRYPTION_KEY") {
        URL_SAFE_NO_PAD.decode(key_base64)?
    } else {
        return Err("No encryption key source configured. Set one of: KMS_KEY_ID, VAULT_KEY_PATH, or ENCRYPTION_KEY".into());
    };

    if key_material.len() != 32 {
        return Err(format!(
            "Invalid key length: expected 32 bytes, got {}",
            key_material.len()
        )
        .into());
    }

    let unbound_key = UnboundKey::new(&AES_256_GCM, &key_material)
        .map_err(|e| format!("Failed to create encryption key: {:?}", e))?;

    let key = LessSafeKey::new(unbound_key);

    ENCRYPTION_KEY
        .set(key)
        .map_err(|_| "Encryption key already initialized")?;

    Ok(())
}

pub fn encrypt_token(token: &str) -> Result<String, Box<dyn std::error::Error>> {
    let key = ENCRYPTION_KEY
        .get()
        .ok_or("Encryption key not initialized")?;

    let rng = SystemRandom::new();
    let mut nonce_bytes = [0u8; 12];
    rng.fill(&mut nonce_bytes)
        .map_err(|e| format!("Failed to generate nonce: {:?}", e))?;

    let mut in_out = token.as_bytes().to_vec();
    in_out.reserve(16);

    let nonce = Nonce::assume_unique_for_key(nonce_bytes);

    let aad_data = format!("slack_token_v1_{}", chrono::Utc::now().timestamp());
    let aad = Aad::from(aad_data.as_bytes());

    key.seal_in_place_append_tag(nonce, aad, &mut in_out)
        .map_err(|e| format!("Encryption failed: {:?}", e))?;

    let mut encrypted = nonce_bytes.to_vec();
    encrypted.extend_from_slice(&in_out);

    Ok(URL_SAFE_NO_PAD.encode(encrypted))
}

// pub fn decrypt_token(encrypted: &str) -> Result<String, Box<dyn std::error::Error>> {
//     let key = ENCRYPTION_KEY
//         .get()
//         .ok_or("Encryption key not initialized")?;

//     let encrypted_bytes = URL_SAFE_NO_PAD
//         .decode(encrypted)
//         .map_err(|e| format!("Invalid base64: {}", e))?;

//     if encrypted_bytes.len() < 12 {
//         return Err("Invalid encrypted data: too short".into());
//     }

//     let (nonce_bytes, ciphertext) = encrypted_bytes.split_at(12);
//     let nonce = Nonce::assume_unique_for_key(*<&[u8; 12]>::try_from(nonce_bytes)?);

//     let mut in_out = ciphertext.to_vec();

//     let aad = Aad::from(b"slack_token_v1_*");

//     key.open_in_place(nonce, aad, &mut in_out).map_err(|e| {
//         error!("Decryption failed - possible tampering detected: {:?}", e);
//         format!("Failed to decrypt token: {:?}", e)
//     })?;

//     String::from_utf8(in_out).map_err(|e| format!("Invalid UTF-8 in decrypted token: {}", e).into())
// }

// #[cfg(feature = "aws")]
// async fn fetch_key_from_kms(key_id: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//     use aws_sdk_kms::{types::DataKeySpec, Client};

//     let config = aws_config::load_from_env().await;
//     let client = Client::new(&config);

//     let response = client
//         .generate_data_key()
//         .key_id(key_id)
//         .key_spec(DataKeySpec::Aes256)
//         .send()
//         .await?;

//     response
//         .plaintext
//         .ok_or("No plaintext key returned from KMS".into())
//         .map(|b| b.into_inner())
// }

#[cfg(not(feature = "aws"))]
fn fetch_key_from_kms(_key_id: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("AWS KMS support not compiled. Enable 'aws' feature".into())
}

#[cfg(not(feature = "vault"))]
fn fetch_key_from_vault(_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("HashiCorp Vault support not compiled. Enable 'vault' feature".into())
}
