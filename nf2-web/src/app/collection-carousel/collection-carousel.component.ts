import { Component, OnInit, Input } from '@angular/core';
import { Store } from '@ngrx/store';
import { Observable } from 'rxjs';

import {GeneratorStore} from '../generate-input/store/types';

@Component({
  selector: 'app-collection-carousel',
  templateUrl: './collection-carousel.component.html',
  styleUrls: ['./collection-carousel.component.css']
})
export class CollectionCarouselComponent implements OnInit {
  @Input() searchTerm: string = '';
  collections$: Observable<GeneratorStore>;
  
  constructor(private store: Store<{ generator: GeneratorStore }>) { 
    this.collections$ = store.select('generator');
  }

  ngOnInit(): void {
  }

}
