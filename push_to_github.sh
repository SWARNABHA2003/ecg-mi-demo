#!/usr/bin/env bash
# First run: creates the GitHub repository and pushes. Later runs: commit + push (the web app updates itself).
#   bash push_to_github.sh "message"         REPO_NAME=ecg-mi-demo  REPO_VISIBILITY=public|private
set -e
cd "$(dirname "$0")"
command -v git >/dev/null || { echo "Install git:  sudo apt install git"; exit 1; }
command -v gh  >/dev/null || { echo "Install the GitHub CLI:  sudo apt install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login --web --git-protocol https
gh auth setup-git >/dev/null 2>&1 || true
[ -d .git ] || { git init -q; git checkout -q -b main; }
if [ -z "$(git config user.name)" ]; then
  login=$(gh api user -q .login)
  git config user.name "$login"
  git config user.email "$login@users.noreply.github.com"
fi
git add -A
git commit -qm "${1:-Update ECG MI demo app}" || echo "Nothing new to commit."
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  gh repo create "${REPO_NAME:-ecg-mi-demo}" --"${REPO_VISIBILITY:-public}" --source=. --remote=origin --push
fi
echo "Pushed: $(gh repo view --json url -q .url)"
