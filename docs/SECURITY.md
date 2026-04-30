# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Homerun, please report it responsibly.

**Do not open a public issue.** Instead, email the maintainer directly or use GitHub's private vulnerability reporting feature.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix**: Depends on severity, but critical issues are prioritized

## Scope

The following are in scope:

- Backend API endpoints (injection, authentication bypass, etc.)
- Private key or credential exposure
- WebSocket vulnerabilities
- Dependencies with known CVEs

The following are out of scope:

- Polymarket platform vulnerabilities (report those to Polymarket directly)
- Issues that require physical access to the host machine
- Social engineering

## Best Practices for Users

- **Never commit secrets** — keep private keys and API secrets out of version control
- **Use shadow trading mode first** — verify the system works against the microstructure-aware fill simulator before enabling live trading
- **Set conservative limits** — use the Settings UI to cap trade sizes and daily volume
- **Keep dependencies updated** — run `pip install --upgrade -r requirements.txt` and `npm update` regularly
- **Run behind a firewall** — the dashboard and API are not designed for public internet exposure without additional authentication
