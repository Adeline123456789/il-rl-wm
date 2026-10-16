# Binary transport preflight

No remote write was attempted. Direct JSON/base64 transport of the 5,734,080-byte committed blob was truncated by the tool output cap despite a larger token request. Reading the exact Git blob in 393,216-byte chunks completed all 15 JSON payloads and produced the expected combined base64 length. The chunk size is divisible by three, so only the final chunk can contain base64 padding.

At batch upload, read committed Git blob bytes (not text-mode file contents), combine the chunks internally, call the authenticated connector, and require its returned Git SHA to match the local blob SHA. This preflight verifies local transport completeness only; it does not verify a GitHub write or claim that artifacts have been delivered. Preserve file modes and compare the final remote tree SHA to the committed local tree.

The first attempt to persist the report used JSON booleans as Python literals and failed before writing the report; corrected by parsing JSON. No source data or remote content was changed.
