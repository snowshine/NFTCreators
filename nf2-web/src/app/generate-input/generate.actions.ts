import { createAction } from '@ngrx/store';

export const startGenerator = createAction('[Generate Asset] Start Generator');
export const pollGenerator = createAction('[Generate Asset] Poll Generator');
export const retrieveAsset = createAction('[Generate Asset] Retrieve Asset');