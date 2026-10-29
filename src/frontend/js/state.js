// Shared application state
const AppState = (() => {
    let currentConfigs = [];
    let localModels = [];
    let expandedConfigs = new Set();
    let expandedStorage = new Set();
    let repoDebounceTimer = null;
    let lastConfigStr = "";
    let lastStorageStr = "";
    let lastCommitsStr = "";
    let selectedBranch = "main";
    let isRpcMode = false;
    let gpuInfo = { gpus: [], radiance: { enabled: false, gpu: "all" } };
    let lastQuantData = null;

    return {
        get gpuInfo() { return gpuInfo; },
        set gpuInfo(val) { gpuInfo = val; },
        get lastQuantData() { return lastQuantData; },
        set lastQuantData(val) { lastQuantData = val; },
        get currentConfigs() { return currentConfigs; },
        set currentConfigs(val) { currentConfigs = val; },
        get localModels() { return localModels; },
        set localModels(val) { localModels = val; },
        get expandedConfigs() { return expandedConfigs; },
        get expandedStorage() { return expandedStorage; },
        get repoDebounceTimer() { return repoDebounceTimer; },
        set repoDebounceTimer(val) { repoDebounceTimer = val; },
        get lastConfigStr() { return lastConfigStr; },
        set lastConfigStr(val) { lastConfigStr = val; },
        get lastStorageStr() { return lastStorageStr; },
        set lastStorageStr(val) { lastStorageStr = val; },
        get lastCommitsStr() { return lastCommitsStr; },
        set lastCommitsStr(val) { lastCommitsStr = val; },
        get selectedBranch() { return selectedBranch; },
        set selectedBranch(val) { selectedBranch = val; },
        get isRpcMode() { return isRpcMode; },
        set isRpcMode(val) { isRpcMode = val; },
        getDefaultParams() {
            return {
                "no-mmap": true,
                "fit": "on",
                "fitc": 16384,
                "fa": "on",
                "cache-type-k": "q8_0",
                "cache-type-v": "q8_0",
                "t": 8,
                "n": -1,
                "temp": 0.6,
                "top-p": 0.95,
                "top-k": 40,
                "min-p": 0
            };
        },
        // radiance's own flags, not llama-server's: one card, fp8 KV cache, 200K context.
        getRadianceDefaultParams() {
            return {
                "tp": 1,
                "max-num-seqs": 8,
                "max-model-len": 200000,
                "kv-cache-dtype": "fp8"
            };
        }
    };
})();
