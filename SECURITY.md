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

- **Never commit `.env` files** — they contain private keys and API secrets
- **Use paper trading mode first** — verify the system works before enabling live trading
- **Set conservative limits** — use `MAX_TRADE_SIZE_USD` and `MAX_DAILY_TRADE_VOLUME` to cap exposure
- **Keep dependencies updated** — run `pip install --upgrade -r requirements.txt` and `npm update` regularly
- **Run behind a firewall** — the dashboard and API are not designed for public internet exposure without additional authentication
