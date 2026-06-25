# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Unlimited-OCR-ROCm, please do **not** open a public issue.

Instead, report it privately via one of these channels:

1. **GitHub Security Advisory:** Go to [Security > Advisories](https://github.com/AIwork4me/Unlimited-OCR-ROCm/security/advisories/new)
2. **Email:** Contact the maintainer directly (see [CODEOWNERS](.github/CODEOWNERS))

We will acknowledge your report within 48 hours and aim to publish a fix within 7 days.

## Scope

- Input parsing and file handling (PDFs, images)
- SGLang server communication
- Environment variable handling
- Dependency chain vulnerabilities

## Out of Scope

- Vulnerabilities in upstream dependencies (please report those to the respective projects)
- Denial-of-service via excessive resource consumption (this is mitigated by SGLang's built-in limits)
