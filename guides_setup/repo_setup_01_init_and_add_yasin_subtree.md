# Remote Repo Setup (GitHub) + `dicomviewer` as a Subtree

> This guide assumes you’re SSH’d into your **remote Linux server** (where you code via Cursor/VS Code Remote). Replace placeholders like `YOUR_GITHUB_USERNAME` and `your-repo-name`.

## 0) One-time prerequisites

```bash
# Check you have git and openssh
git --version
ssh -V
```

### 0.1) Update Git (if needed for SSH GPG signing)

If your Git version is older than 2.34.0 and you want SSH GPG signing support:  
My original git version is 2.25 which comes with the server.  
I ended up using method 3 on shenggpu8

```bash
# Check current Git version
git --version



# Method 1: Install from Git's official PPA (Ubuntu/Debian)
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:git-core/ppa -y
sudo apt update
sudo apt install git -y

# Method 2: If add-apt-repository fails, manually add PPA
echo "deb http://ppa.launchpad.net/git-core/ppa/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/git-core-ppa.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E1DD270288B4E6030699E45FA1715D88E1DF6F24
sudo apt update
sudo apt install git -y

# Method 3: Compile from source (if PPA methods fail)
sudo apt install -y make libssl-dev libghc-zlib-dev libcurl4-gnutls-dev libexpat1-dev gettext unzip
cd /tmp
wget https://github.com/git/git/archive/v2.43.0.tar.gz
tar -xzf v2.43.0.tar.gz
cd git-2.43.0
make prefix=/usr/local all
sudo make prefix=/usr/local install
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify new Git version
git --version
```

### 0.2) SSH agent setup

```bash
# Optional but helpful: ensure your shell loads ssh-agent automatically
# (If you use bash)
grep -q "ssh-agent" ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# Auto-start ssh-agent and add default keys
if ! pgrep -u "$USER" ssh-agent >/dev/null; then
  eval "$(ssh-agent -s)"
fi
[ -f ~/.ssh/id_ed25519 ] && ssh-add ~/.ssh/id_ed25519 >/dev/null 2>&1 || true
EOF
```

---

## 1) Create SSH key on the server & add to GitHub
Note I should be able to copy the ~.ssh folder to other servers

```bash
# Generate a modern SSH keypair for GitHub authentication
ssh-keygen -t ed25519 -C "hengjie@[remote-server]"   # choose a passphrase when prompted
ssh-keygen -t ed25519 -C "hengjie@10.72.22.84"   # choose a passphrase when prompted
# Then for the location I use default (press Enter), and for passphrase I use xgg style

# Print the public key (copy this to GitHub)
cat ~/.ssh/id_ed25519.pub
```

1. Go to **GitHub → Settings → SSH and GPG keys → New SSH key**.  
2. Paste the **public** key output (the `.pub` content).

```bash
# Test GitHub SSH works (should say "successfully authenticated")
ssh -T git@github.com
```

> If you see a warning about authenticity, type `yes` once to trust GitHub’s host key.

---

## 2) Configure Git identity (scoped to this repo only)

> Keeps machine identities clean. We’ll set global defaults minimal, then **override per repo**.

```bash
# Optional global defaults (used if a repo hasn't set its own)
git config --global user.name  "Hengjie Liu"
git config --global user.email "[YOUR_GITHUB_EMAIL]"
git config --global user.email "hjliu@g.ucla.edu"

# Enable signed commits using your SSH key (nice-to-have)
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

You can override `user.name`/`user.email` **locally** inside the repo later (§4).

---

## 3) Create a new GitHub repo

1. On GitHub, **create a new repository**: `YOUR_GITHUB_USERNAME/your-repo-name`.  
   - Choose **Private** or **Public**.  
   - **Do not** initialize with README (we’ll push our own).

---

## 4) Initialize your repo on the remote server

```bash
# Make a project directory (or use an existing one)
[mkdir -p ~/code/your-repo-name]
[cd ~/code/your-repo-name]
mkdir /home/hengjie/DL_projects/code_sync/myMedIADIRLab
cd /home/hengjie/DL_projects/code_sync/myMedIADIRLab

# Initialize git
git init

# Set per-repo identity (recommended)
git config user.name  "Hengjie Liu"
git config user.email "[YOUR_GITHUB_EMAIL]"
git config --global user.email "hjliu@g.ucla.edu"

# Optional: enforce fast-forward only merges to keep history clean
git config pull.ff only # I chose to use it for now

# Create a basic README
cat > README.md <<'EOF'
# myMedIADIRLab
My repo for medical image analysis with focus on deformable image registraiton
EOF

# (Recommended) .gitignore for DL/MRI projects
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
.venv/
.env

# DL runs & logs
runs/
checkpoints/
outputs/
wandb/
lightning_logs/

# Large artifacts (tracked with LFS if needed)
*.pt
*.pth
*.onnx
*.npz
*.npy
*.ckpt
*.nii
*.nii.gz
*.mha
*.h5
*.tar
*.zip

# Jupyter Lab & Notebooks
.ipynb_checkpoints/
*/.ipynb_checkpoints/*
*.ipynb_checkpoints
.jupyter/
jupyter_notebook_config.py
jupyter_lab_config.py

# Editors & OS
.DS_Store
.vscode/
.idea/

# Docker
*.tar.gz
EOF

# (Optional) Git LFS for models & moderate artifacts (NOT for huge datasets)
git lfs install
git lfs track "*.pt" "*.pth" "*.onnx" "*.ckpt" "*.npy" "*.npz" "*.nii" "*.nii.gz" "*.dat"
git add .gitattributes

# First commit
git add .
git commit -m "init: project skeleton"
```

Add the GitHub remote and push:

```bash
# Use SSH form (recommended)
git remote add origin git@github.com:[YOUR_GITHUB_USERNAME/your-repo-name.git]
git remote add origin git@github.com:HengjieLiu/myMedIADIRLab

# Name your default branch 'main' if not already
git branch -M main

# Push the initial history
git push -u origin main
```

> After pushing, consider protecting `main` in GitHub (Settings → Branches → Add rule): require PRs & CI checks.

---

## 5) (Optional/Skipped) Pre-commit hooks (auto format & lint)

```bash
# Install tooling in your Python env
pip install pre-commit black isort flake8

# Add config
cat > .pre-commit-config.yaml <<'YAML'
repos:
- repo: https://github.com/psf/black
  rev: 24.8.0
  hooks: [{id: black}]
- repo: https://github.com/pycqa/isort
  rev: 5.13.2
  hooks: [{id: isort}]
- repo: https://github.com/pycqa/flake8
  rev: 7.1.1
  hooks: [{id: flake8}]
YAML

# Enable hooks
pre-commit install

# First run across the repo (formats existing files)
pre-commit run --all-files

# Commit any changes the hooks made
git add -A
git commit -m "chore: add pre-commit (black, isort, flake8)"
git push
```

---

## 6) Add `dicomviewer` (GitLab) as a **subtree**

> We will import `https://gitlab.com/YAAF/dicomviewer` under `third_party/dicomviewer/`.  
> **Subtree** advantages: appears as a normal folder, easy to edit, easy to pull from upstream, and still possible to contribute back later.

### 6.1 Add the upstream remote and import

```bash
# Add the GitLab project as an *extra remote* named 'dicomviewer'
# Note: Use HTTPS instead of SSH if you don't have GitLab SSH key set up
git remote add dicomviewer https://gitlab.com/YAAF/dicomviewer.git

# If git subtree command is not recognized, install git-all first: (I actually used this)
sudo apt install git-all -y

# If git subtree still not available, use direct path: (I actually used this)
/usr/lib/git-core/git-subtree add --prefix=third_party/dicomviewer dicomviewer main --squash

# Option A: Import preserving full history into a subfolder (heavy)
# git subtree add --prefix=third_party/dicomviewer dicomviewer main

# Option B: Import with --squash (keeps your repo lighter; recommended)
# Make sure working tree is clean first: git status
git subtree add --prefix=third_party/dicomviewer dicomviewer main --squash

# Commit message is auto-generated; push to GitHub
# If GPG signing fails, use: git push --no-gpg-sign
git push
```

> The code now lives at `third_party/dicomviewer/` in **your** repo—no submodule headaches.

### 6.2 Pull upstream updates later

```bash
# Fetch latest from GitLab remote
git fetch dicomviewer

# Merge updates into your subtree (again using --squash to keep history compact)
git subtree pull --prefix=third_party/dicomviewer dicomviewer main --squash

# Push to GitHub
git push
```

### 6.3 Work on `dicomviewer` inside your repo

```bash
# Edit files directly under third_party/dicomviewer/
# Stage, commit, and push as usual
git add third_party/dicomviewer
git commit -m "dicomviewer: fix XYZ for our pipeline"
git push
```

---

## 7) Contribute changes back to the GitLab project (later)

> When you have upstream-worthy changes, split your subtree history into its own branch and push it to a **fork** on GitLab. Then open a Merge Request.

```bash
# Create a branch containing the history of the subtree path
git subtree split --prefix=third_party/dicomviewer -b dicomviewer-split

# (One-time) Fork "YAAF/dicomviewer" on GitLab to your own namespace.
# Then add your fork as a remote; replace YOUR_GITLAB_USERNAME below.
git remote add dicomviewer-fork git@gitlab.com:YOUR_GITLAB_USERNAME/dicomviewer.git

# Push the split branch to your fork (use a feature branch name)
git push dicomviewer-fork dicomviewer-split:feature/your-change

# Open a Merge Request on GitLab from your fork's feature branch -> upstream main
```

**Tip:** Keep a small file to document this flow:

```bash
mkdir -p third_party/dicomviewer
cat > third_party/dicomviewer/UPSTREAM.md <<'EOF'
# dicomviewer (upstream)

- Upstream: https://gitlab.com/YAAF/dicomviewer
- Imported: branch `main`, via `git subtree add --prefix=third_party/dicomviewer dicomviewer main --squash`
- Update: `git fetch dicomviewer` then `git subtree pull --prefix=third_party/dicomviewer dicomviewer main --squash`
- Contribute back:
  git subtree split --prefix=third_party/dicomviewer -b dicomviewer-split
  git push git@gitlab.com:YOUR_GITLAB_USERNAME/dicomviewer.git dicomviewer-split:feature/your-change
  # Open MR from your fork to upstream
EOF

git add third_party/dicomviewer/UPSTREAM.md
git commit -m "docs: add UPSTREAM.md for dicomviewer subtree"
git push
```

> **License note:** Before redistributing or upstreaming, confirm `dicomviewer`’s license terms in its repository. If redistribution is restricted, keep it private and upstream only patches you authored.

---

## 8) Daily workflow tips

```bash
# Keep main updated (fast-forward only)
git checkout main
git pull --ff-only

# Start a feature branch for your change
git checkout -b feat/some-improvement

# Work, commit, push
git add -p
git commit -s -m "feat: improve XYZ"
git push -u origin feat/some-improvement

# Open a Pull Request on GitHub
```

---

## 9) Optional: Lightweight CI (GitHub Actions)

```bash
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'YAML'
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - run: pip install -r requirements.txt || true
      - run: pip install -U pytest || true
      - run: pytest -q || true
YAML

git add .github/workflows/ci.yml
git commit -m "ci: add minimal GitHub Actions workflow"
git push
```

> Keep tests CPU-only and fast; skip GPU/huge data with pytest markers.

---

## 10) Docker notes (if you containerize)

- Bind-mount your repo rather than committing inside the container:
  ```bash
  docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace yourimage
  ```
- If you run git **inside** a container (root user), mark the repo safe:
  ```bash
  git config --global --add safe.directory /workspace
  ```
- Add a `.dockerignore` mirroring your `.gitignore` plus large local dirs to keep images slim.

---

## 11) Troubleshooting

- **`ssh -T git@github.com` asks to confirm host**: type `yes` once.  
- **Permission denied (publickey)**: ensure your key is added to ssh-agent and to GitHub:
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```
- **Subtree pull merge conflicts**: resolve files under `third_party/dicomviewer/`, then:
  ```bash
  git add -A
  git commit
  git push
  ```
- **Accidentally added big data**: remove with `git rm --cached -r data/` and commit; consider DVC or external storage.

---

## 12) Quick reference (subtree)

```bash
# Initial add
git remote add dicomviewer git@gitlab.com:YAAF/dicomviewer.git
git subtree add --prefix=third_party/dicomviewer dicomviewer main --squash

# Update later
git fetch dicomviewer
git subtree pull --prefix=third_party/dicomviewer dicomviewer main --squash

# Contribute back
git subtree split --prefix=third_party/dicomviewer -b dicomviewer-split
git push git@gitlab.com:YOUR_GITLAB_USERNAME/dicomviewer.git dicomviewer-split:feature/your-change
```
