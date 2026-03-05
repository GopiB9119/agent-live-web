# Release Checklist

Use this checklist before publishing any new release tag.

## 1) Required checks are green
- Confirm PR checks passed:
  - `test-python-agent`
  - `evaluate`
- Confirm branch protection is still strict:
  - required review count = `1`
  - code owner reviews = `true`
  - conversation resolution = `true`

## 2) Release tag is created and pushed
- Create annotated tag:
  - `git tag -a vX.Y.Z -m "vX.Y.Z - <summary>"`
- Push tag:
  - `git push origin vX.Y.Z`
- Create GitHub release notes for the same tag.

## 3) Sigstore attestation verification

Use one of the following verification paths.

### Option A: GitHub attestation verify
- Verify artifact attestation against this repo:
  - `gh attestation verify <artifact-path> --repo GopiB9119/agent-live-web`

### Option B: Cosign bundle verify
- Verify a downloaded `.sigstore.json` bundle:
  - `cosign verify-blob <artifact-path> --bundle <attestation.sigstore.json> --certificate-oidc-issuer https://token.actions.githubusercontent.com --certificate-identity-regexp "https://github.com/GopiB9119/agent-live-web/.*"`

## 4) Post-release sanity
- Confirm release page is public and assets are accessible.
- Confirm latest tag appears in `git ls-remote --tags origin`.
- Record release notes link in project updates/changelog.
