# Security Policy

## Supported versions

Security fixes are applied to the latest commit on `main`. TorchForge does not
currently maintain older release branches.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting for this repository:

1. Open the repository's **Security** tab.
2. Select **Advisories**.
3. Select **Report a vulnerability**.

Include the affected version or commit, reproduction steps, impact, and any
suggested mitigation. If private vulnerability reporting is unavailable,
contact the repository owner privately through their GitHub profile.

You can expect an initial acknowledgement within seven days. Please allow time
for a fix and coordinated disclosure before publishing details.

## Scope

Reports involving generated-code execution, path traversal, unsafe artifact
handling, dependency vulnerabilities, or unintended network exposure are
especially useful.

TorchForge's AST validation is a correctness and risk-reduction layer, not a
security sandbox. Run untrusted generated code inside an isolated account,
container, or virtual machine.

## Known accepted risks

- `image-size` (a transitive dependency of the frontend build tool `vinext`)
  has unpatched denial-of-service advisories in its ICNS/JXL/HEIF parsers
  ([GHSA-w3rx-r6r6-pgpr](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr),
  [GHSA-5p2g-fcmc-qvqq](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq)).
  It runs at build time on local files inside this repository only, so the risk
  is considered negligible. No patched upstream release exists yet; this will
  be resolved when `vinext` or `image-size` ships a fix.
