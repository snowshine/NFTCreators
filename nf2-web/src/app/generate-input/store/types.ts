export enum GenerationStatus {
    NOT_STARTED = "NOT_STARTED",
    STARTED = "STARTED",
    PENDING = "PENDING",
    PROCESSING = "PROCESSING",
    COMPLETE = "COMPLETE",
    FAILED = "FAILED"    
}

export interface GeneratorStore {
    status: GenerationStatus;
    collections: string[],
    collection: string|null,
    jobId: string|null;
    jobStatus: string|null;
    asset: string|null;
}
