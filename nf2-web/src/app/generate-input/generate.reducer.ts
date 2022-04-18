import { createReducer, on, State } from '@ngrx/store';
import { startGenerator, pollGenerator, retrieveAsset } from './generate.actions';

export interface GENERATOR_STORE {
    status: string;
    asset: string | null;
}

export const initialState: GENERATOR_STORE = {
    status: 'NOT STARTED',
    asset: null,
};

export const generateReducer = createReducer(
  initialState,
  on(startGenerator, (state: GENERATOR_STORE) => { return { ...state, status: 'STARTED'} }),
);