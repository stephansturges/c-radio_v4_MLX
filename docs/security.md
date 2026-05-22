# Security and Provenance

Bundle metadata must preserve:

- source model ID
- source revision SHA
- conversion package version
- dtype and quantization metadata
- upstream license/provenance notes
- file hashes once the weight writer lands

The runtime should load local safetensors and local metadata only. Remote code is allowed only
during controlled reference conversion from pinned revisions.
