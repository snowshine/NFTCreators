import {Injectable} from '@angular/core';
import {Actions, Effect, ofType, ROOT_EFFECTS_INIT} from '@ngrx/effects';
import {HttpClient} from '@angular/common/http';
import { Store } from '@ngrx/store';
import {map, catchError, switchMap, withLatestFrom} from 'rxjs';
import { delay } from 'rxjs/operators';

import {GeneratorStore} from './types'
import {GenerateTypes, TokenTypes, LoadCollectionsSuccess, GenerationStartSuccess, GenerationStartFailure, GenerateToken, GenerateTokenSuccess, GenerateTokenError} from './generate.actions';

const mockApi = 'http://virtserver.swaggerhub.com/jasoncoffman/NF2/1.0.0';
const apiGateway = 'https://53p31mjl34.execute-api.us-east-2.amazonaws.com/dev';
const activeApi = apiGateway;

@Injectable()
export class NewsEffects {

    constructor(
        private actions$: Actions, 
        private http: HttpClient,
        private store: Store<{ generator: GeneratorStore }>
    ) {}

    @Effect()
    loadCollections = this.actions$.pipe(
        ofType(ROOT_EFFECTS_INIT),
        switchMap((action: any) => {
            return this.http.get(`${activeApi}/collections`).pipe(
            map((response: any) => LoadCollectionsSuccess({ collections: response }))
            );
        }),
        );

    @Effect()
    generateToken = this.actions$.pipe(
    ofType(GenerateTypes.Submit),
    withLatestFrom(this.store.select('generator')),
    switchMap(([action, latest]) => {
        return this.http.put(`${activeApi}/generate?collectionSlug=${latest.collection}`, {}).pipe(
        map((response: any) => {
            if (response.jobId) {
                return GenerationStartSuccess({jobId: response.jobId});
            } else {
                return GenerationStartFailure();
            }
        })
        );
    }),
    );

    @Effect()
    pollToken = this.actions$.pipe(
    ofType(
        GenerateTypes.SubmitSuccess,
        TokenTypes.Generate,
    ),
    delay(2500),
    withLatestFrom(this.store.select('generator')),
    switchMap(([action, latest]) => {
        return this.http.get(`${activeApi}/generate/${latest.jobId}?collectionSlug=${latest.collection}`, {}).pipe(
    map((response: any) => {
        if (response.Status === 'COMPLETE') {
            return GenerateTokenSuccess({assetLocation: response.SignedLocation});
        } else if (response.Status === 'FAILED') {
            return GenerateTokenError();
        }

        return GenerateToken({jobStatus: response.Status});
    }));
    }),
    );
}