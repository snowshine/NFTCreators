import { createFeatureSelector, createSelector } from '@ngrx/store';
import {GeneratorStore} from './types';

export const getGenerateStore = createFeatureSelector<GeneratorStore>('generator');

export const getAssetLocation = createSelector(
    getGenerateStore,
    (state: GeneratorStore) => state.asset
);