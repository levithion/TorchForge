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
