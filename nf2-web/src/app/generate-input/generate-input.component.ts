import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Store, select } from '@ngrx/store';
import { Observable } from 'rxjs';

import {trait_graph} from './graphs/traits';
import {sales_graph} from './graphs/sales';
import {sales_stat} from './graphs/sales_stats';
import {GeneratorStore} from './store/types'
import {SubmitGenerate, ClearGenerate, SetCollection} from './store/generate.actions';
import {getAssetLocation} from './store/generate.selectors';

@Component({
  selector: 'app-generate-input',
  templateUrl: './generate-input.component.html',
  styleUrls: ['./generate-input.component.css']
})
export class GenerateInputComponent implements OnInit {
  slugId: string = '';
  assetURL: string = '';
  status$: Observable<GeneratorStore>;
  assetLocation$: Observable<string|null>;
  isButtonVisible: boolean = true;

  trait_data: any[] = [];
  sales_data: any[] = [];
  stats: any = {};
  view: [number, number] = [400, 200];

  // options
  showXAxis: boolean = true;
  showYAxis: boolean = true;
  gradient: boolean = true;
  showLegend: boolean = true;
  showXAxisLabel: boolean = true;
  xAxisLabel: string = 'Log price of token (USD)';
  showYAxisLabel: boolean = true;
  yAxisLabel: string = 'Number of tokens';
  legendTitle: string = 'Log price of token (USD)';

  constructor(
    private route: ActivatedRoute,
    private store: Store<{ generator: GeneratorStore }>
  ) {
    this.store = store;
    this.status$ = store.select('generator');
    this.assetLocation$ = this.store.pipe(select(getAssetLocation));

    this.route.params.subscribe((params: any) => {
      this.slugId = params.slug.toLowerCase();
      this.store.dispatch(SetCollection({collection: params.slug.toLowerCase()}));
      this.trait_data = trait_graph[params.slug.toLowerCase()];
      this.sales_data = sales_graph[params.slug.toLowerCase()] ? sales_graph[params.slug.toLowerCase()].map((data: any) => { return {
        ...data, value: (data["value"] ? Math.log10(data["value"]) : 0)
      }}) : null;
      this.stats = sales_stat[params.slug.toLowerCase()];
    });

    this.assetLocation$.pipe()
      .subscribe((asset: string | null) => {
        this.assetURL = asset || '';
    });
  }

  startGenerator(): void {
    this.store.dispatch(SubmitGenerate());
  }

  resetGenerator(): void {
    this.store.dispatch(ClearGenerate())
  }

  downloadAsset(): void {
    window.open(this.assetURL,'_blank')
  }
  
  ngOnInit(): void {
    const header = document.getElementById('navigation-header');
    if (header) header.style.boxShadow = "0 4px 4px rgba(0,0,0,0.15), 0 8px 8px rgba(0,0,0,0.15)";
  }

  ngOnDestroy(): void {
    this.store.dispatch(ClearGenerate());
    const header = document.getElementById('navigation-header');
    if (header) header.style.boxShadow = "";
  }

}
