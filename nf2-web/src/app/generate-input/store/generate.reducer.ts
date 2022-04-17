import { createReducer, on, State } from '@ngrx/store';
import { SubmitGenerate, ClearGenerate, GenerateToken, pollGenerator, GenerationStartSuccess, LoadCollectionsSuccess, GenerateTokenSuccess, SetCollection } from './generate.actions';
import {GenerationStatus, GeneratorStore} from './types';

export const initialState: GeneratorStore = {
  status: GenerationStatus.NOT_STARTED,
  collections: [],
  collection: null,
  jobId: null,
  jobStatus: null,
  asset: null,
};

export const generateReducer = createReducer(
  initialState,
  on(SetCollection, (state: GeneratorStore, payload: any) => { return {...state, collection: payload.collection}}),
  on(ClearGenerate, (state: GeneratorStore) => { return {...initialState, collections: state.collections, collection: state.collection}}),
  on(SubmitGenerate, (state: GeneratorStore) => { return { ...state, status: GenerationStatus.STARTED} }),
  on(GenerateToken, (state: GeneratorStore, payload: any) => { return { ...state, jobStatus: payload.jobStatus} }),
  on(LoadCollectionsSuccess, (state: GeneratorStore, payload: any) => {
    return {...state, collections: payload.collections};
  }),
  on(GenerationStartSuccess, (state: GeneratorStore, payload: any) => {
    return {...state, jobId: payload.jobId, jobStatus: payload.jobStatus};
  }),
  on(GenerateTokenSuccess, (state: GeneratorStore, payload: any) => {
    return {...state, asset: payload.assetLocation};
  },)
);