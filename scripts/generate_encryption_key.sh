#!/bin/bash

# Generate a secure 32-byte (256-bit) encryption key for Interstice Engine
# This key is used for encrypting sensitive data like Slack tokens

echo "Generating 32-byte encryption key for Interstice Engine..."
echo ""

# Generate 32 random bytes and encode as base64
ENCRYPTION_KEY=$(openssl rand -base64 32)

echo "Add this to your .env file:"
echo "ENCRYPTION_KEY=$ENCRYPTION_KEY"
echo ""
echo "Or run this command to add it automatically:"
echo "echo 'ENCRYPTION_KEY=$ENCRYPTION_KEY' >> .env"
echo ""
echo "⚠️  IMPORTANT: Keep this key secure and never commit it to version control!"
echo "   This key is used to encrypt sensitive data in your database."
