#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup-github-secrets.sh
#
# Run this in Azure Cloud Shell (https://shell.azure.com) to print all the
# values you need to paste into GitHub → Settings → Secrets → Actions.
#
# Usage:
#   chmod +x scripts/setup-github-secrets.sh
#   bash scripts/setup-github-secrets.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

RESOURCE_GROUP="customer-intel-rg2"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Customer Intelligence Platform — GitHub Secrets Helper"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── 1. Find the ACR in this resource group ───────────────────────────────────
echo "🔍 Finding Azure Container Registry in $RESOURCE_GROUP ..."
ACR_NAME=$(az acr list \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].name" -o tsv)

if [ -z "$ACR_NAME" ]; then
  echo "❌ No ACR found in $RESOURCE_GROUP. Please check your resource group name."
  exit 1
fi

ACR_SERVER=$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "loginServer" -o tsv)

echo "✅ ACR found: $ACR_NAME ($ACR_SERVER)"
echo ""

# ── 2. Get ACR credentials ────────────────────────────────────────────────────
echo "🔑 Fetching ACR credentials ..."
az acr update --name "$ACR_NAME" --admin-enabled true --output none

ACR_USER=$(az acr credential show \
  --name "$ACR_NAME" \
  --query "username" -o tsv)

ACR_PASS=$(az acr credential show \
  --name "$ACR_NAME" \
  --query "passwords[0].value" -o tsv)

# ── 3. Create service principal for GitHub Actions ────────────────────────────
echo "🤖 Creating service principal for GitHub Actions ..."
SUBSCRIPTION_ID=$(az account show --query "id" -o tsv)

SP_JSON=$(az ad sp create-for-rbac \
  --name "github-customer-intel-frontend" \
  --role contributor \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  --sdk-auth 2>/dev/null)

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COPY THESE VALUES INTO GITHUB SECRETS"
echo "  (Repo → Settings → Secrets and variables → Actions)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Secret name:  AZURE_REGISTRY_LOGIN_SERVER"
echo "Secret value: $ACR_SERVER"
echo ""
echo "Secret name:  AZURE_REGISTRY_USERNAME"
echo "Secret value: $ACR_USER"
echo ""
echo "Secret name:  AZURE_REGISTRY_PASSWORD"
echo "Secret value: $ACR_PASS"
echo ""
echo "Secret name:  AZURE_RESOURCE_GROUP"
echo "Secret value: $RESOURCE_GROUP"
echo ""
echo "Secret name:  GEMINI_API_KEY"
echo "Secret value: <paste your key from .env>"
echo ""
echo "Secret name:  AZURE_CREDENTIALS"
echo "Secret value (paste the entire JSON block below):"
echo ""
echo "$SP_JSON"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done! After adding all 6 secrets, run:"
echo "  git push origin main  (or trigger workflow manually)"
echo "════════════════════════════════════════════════════════════"
