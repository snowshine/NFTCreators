import { Component, OnInit } from '@angular/core';
import { Store } from '@ngrx/store';
import { Observable } from 'rxjs';

import {GeneratorStore} from '../generate-input/store/types';

@Component({
  selector: 'app-generate-token',
  templateUrl: './generate-token.component.html',
  styleUrls: ['./generate-token.component.css']
})
export class GenerateTokenComponent implements OnInit {
  store$: Observable<GeneratorStore>;

  constructor(
    private store: Store<{ generator: GeneratorStore }>
  ) {
    this.store$ = store.select('generator');
  }

  ngOnInit(): void {
  }

}
