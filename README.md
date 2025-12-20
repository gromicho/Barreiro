# Address Persistence and Versioning (Dropbox-backed)

This repository contains the persistence layer and UI integration for
versioned address storage backed by Dropbox.

Each update to the address list creates an immutable, versioned snapshot,
allowing users to inspect, load, and compare historical address sets.

---

## Directory Structure

Address data is stored per instance and versioned incrementally.

```
instances/
└── <instance_name>/
    ├── addresses.1.json
    ├── addresses.2.json
    ├── addresses.3.json
    └── addresses.json
```

- `<instance_name>` is derived from the logical filename (for example `capelle`)
- Each `addresses.<version>.json` file is an immutable snapshot
- `addresses.json` always points to the most recent version

---

## Snapshot Payload Format

Each versioned snapshot is stored as a JSON document with the following
structure:

```json
{
  "version": 12,
  "updated_at": "2025-12-17T17:11:00Z",
  "addresses_text": "...",
  "address_count": 1837,
  "geocoding_cache": {}
}
```

### Fields

| Field | Description |
|------|-------------|
| `version` | Monotonic version number |
| `updated_at` | UTC timestamp of snapshot creation |
| `addresses_text` | Raw address input, one address per line |
| `address_count` | Number of non-empty address lines |
| `geocoding_cache` | Cached geocoding results |

The `address_count` field is computed once at save time and persisted with
the snapshot to ensure consistency across versions.

---

## Version Listing

Available versions are discovered via Dropbox folder metadata and enriched
with snapshot information.

```python
list_address_versions(filename='capelle')
```

Returns a list of dictionaries with the following keys:

- `version` (int)
- `timestamp` (str, UTC, derived from Dropbox metadata)
- `path` (Dropbox path)
- `address_count` (int or None)

Older snapshots created before `address_count` was introduced return `None`
for this field.

---

## UI Integration

The UI version selector displays version information using the persisted
metadata:

```
v12 (1837 addresses, 2025-12-17 17:11 UTC)
v11 (1794 addresses, 2025-12-16 09:42 UTC)
v10 (2025-12-15 14:03 UTC)
```

This allows users to:

- See growth over time
- Detect accidental resets
- Load a specific historical snapshot

---

## Design Principles

- **Immutable snapshots**  
  Versions are append-only and never modified.

- **Single source of truth**  
  Address counts are derived from persisted data, not recomputed.

- **Dropbox-safe**  
  Only small JSON files are downloaded when listing versions.

- **Backward compatible**  
  Older snapshots remain readable without migration.

- **Minimal coupling**  
  UI code consumes metadata without influencing persistence logic.

---

## Future Extensions (Not Implemented)

Potential extensions, deliberately deferred:

- Persisting `geocoded_count` alongside `address_count`
- Maintaining a lightweight `versions.json` index
- Adding monotonicity checks for version sanity

These can be added without breaking existing snapshots.

---

## Notes

This repository focuses strictly on persistence and version management.
Notebooks, analysis code, and UI logic are considered consumers of this API
and are intentionally kept separate.
