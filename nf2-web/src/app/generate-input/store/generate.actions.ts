import { createAction, props } from '@ngrx/store';

export enum CollectionsTypes {
    Load = '[Collections] LOAD',
    LoadSuccess = '[Collections] LOAD SUCCESS',
    LoadError = '[Collections] LOAD ERROR',
}

export enum GenerateTypes {
    Clear = '[Generate] CLEAR',
    Submit = '[Generate] SUBMIT',
    SubmitSuccess = '[Generate] SUBMIT SUCCESS',
    SubmitError = '[Generate] SUBMIT ERROR',
}

export enum TokenTypes { 
    Generate = '[Token] GENERATE',
    GenerateSuccess = '[Token] GENERATE SUCCESS',
    GenerateError = '[Token] GENERATE ERROR',
}

export const SetCollection = createAction('[Collection] SET', props<{collection: string;}>(),);
export const LoadCollectionsSuccess = createAction(CollectionsTypes.LoadSuccess, props<{collections: string[];}>(),);

export const ClearGenerate = createAction(GenerateTypes.Clear)
export const SubmitGenerate = createAction(GenerateTypes.Submit);
export const GenerationStartSuccess = createAction(GenerateTypes.SubmitSuccess, props<{jobId: string;}>(),);
export const GenerationStartFailure = createAction(GenerateTypes.SubmitError);

export const pollGenerator = createAction('[Generate Asset] Generator Started');
export const retrieveAsset = createAction('[Generate Asset] Retrieve Asset');

export const GenerateToken = createAction(TokenTypes.Generate, props<{jobStatus: string;}>(),);
export const GenerateTokenSuccess = createAction(TokenTypes.GenerateSuccess, props<{assetLocation: string;}>(),);
export const GenerateTokenError = createAction(TokenTypes.GenerateError);