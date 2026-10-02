# Onboarding media manifest

Screenshots, GIFs, videos, and report assets stay outside Git. This text
manifest defines scenes and acceptance checks so maintainers can reproduce
public media without committing binary artifacts.

| ID | Scene | Required visible proof | Secret treatment |
| --- | --- | --- | --- |
| `first-turn` | `codeflow onboard --skip-memory` from language selection through the first reply | Provider selected, Memory explicitly disabled, and an `Agent:` reply | API key masked; disposable repository path |
| `feishu-config` | Feishu Open Platform plus CodeFlow CLI | long connection, `im.message.receive_v1`, published version, and redacted channel config | App Secret, Encrypt Key, Verification Token, and tenant identity redacted |
| `feishu-live` | one inbound Feishu message and the CodeFlow reply | accepted inbound event and reply in the same conversation | user names, open IDs, message IDs, and credentials redacted |
| `agent-install` | released CodeFlow installation and JSON health check | installed version plus `codeflow doctor --json` | GitHub Token, signed URL query strings, and local home paths redacted |

## Capture rules

- Capture the released wheel installation, not an editable developer checkout.
- Bind every asset to a CodeFlow tag, operating system, and capture date in the
  external asset description.
- Use a disposable Git repository and disposable Provider or Feishu
  credentials.
- Keep commands and output readable at README width. Crop terminal chrome
  instead of shrinking text.
- A skipped probe, fixture, or simulated bot reply cannot be labelled as a live
  success.
- Publish assets only to a maintainer-approved external location. Add README
  links only after the URLs are stable and readable by the intended audience.
