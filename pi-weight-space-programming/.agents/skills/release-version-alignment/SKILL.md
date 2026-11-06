---
name: release-version-alignment
description: "Guidelines and step-by-step checklist to align project versions across configuration files, lockfiles, UI headers, and engine metadata before creating git tags or publishing GitHub releases."
license: Apache-2.0
---

# Release Version Alignment & Tagging Checklist

Before creating any git tag or publishing a release, the agent **MUST** ensure that all project configuration, lockfiles, and source files are aligned to the target release version number in a dedicated commit or PR.

Never create a release or git tag without first updating and verifying the version across all 5 key locations in this repository.

---

## 1. Version-Bearing Locations

Every release requires updating the version in the following 5 files:

### 1. `package.json`
Update the top-level version field:
```json
{
  "name": "pi-weight-space-programming",
  "version": "X.Y.Z"
}
```

### 2. `pyproject.toml`
Update the project table version:
```toml
[project]
name = "pi-weight-space-programming"
version = "X.Y.Z"
```

### 3. `uv.lock`
Regenerate and sync the lockfile to the updated `pyproject.toml` version:
```bash
uv lock
```
> [!IMPORTANT]
> Always verify that `uv.lock` and any `*.lock` files are tracked in git and never ignored by `.gitignore`.

### 4. `extensions/header.ts`
Update the banner title and version display string shown in the Pi extension header:
```typescript
const title = chalk.bold.cyan("Pi Weight-Space Programming vX.Y.Z");
```

### 5. `harness/wsp_engine.py`
Update the default Hugging Face download client User-Agent string:
```python
headers = {
    "User-Agent": "pi-weight-space-programming/X.Y.Z",
    ...
}
```

---

## 2. Pre-Release Verification Checklist

Before committing or pushing the release changes:

1. **Stale Version Check**:
   Search the entire repository for occurrences of the previous version number to ensure no references were missed:
   ```bash
   git grep "<old-version>"
   ```

2. **Gitignore & Tracked Files Check**:
   Ensure essential artifact files are not accidentally ignored:
   - Lockfiles (`uv.lock`) must be tracked.
   - Execution session transcripts (`session.log` in example directories) must be tracked.
   ```bash
   git check-ignore -v uv.lock
   git check-ignore -v examples/model-poisoning/session.log
   ```

3. **Run Functional Tests**:
   Ensure all tests pass with the updated dependencies and lockfiles:
   ```bash
   uv run evals/functional/test_functional.py
   ```

---

## 3. Release Execution Workflow

Follow this sequence to publish the release:

1. **Commit Changes**:
   Stage and commit all version alignment edits:
   ```bash
   git add package.json pyproject.toml uv.lock extensions/header.ts harness/wsp_engine.py
   git commit -m "chore(release): bump version to X.Y.Z"
   ```

2. **Push Branch & Open PR**:
   Push to the feature/release branch and open a pull request for review.
   ```bash
   git push origin <branch-name>
   gh pr create --title "chore(release): bump version to X.Y.Z" --body "..."
   ```

3. **Wait for Approval**:
   > [!WARNING]
   > Do **NOT** merge the PR, push tags, or create a release without explicit user confirmation.

4. **Merge PR & Pull Main**:
   Once approved:
   ```bash
   gh pr merge <pr-number> --merge
   git checkout main
   git pull origin main
   ```

5. **Tag & Publish Release**:
   Tag the commit on `main` and publish via GitHub CLI:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
   ```
