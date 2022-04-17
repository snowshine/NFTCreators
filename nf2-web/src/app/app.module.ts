import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { HttpClientModule, HttpClient } from '@angular/common/http';
import { StoreModule } from '@ngrx/store';
import { EffectsModule } from '@ngrx/effects';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { MarkdownModule } from 'ngx-markdown';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { CollectionCardComponent } from './collection-card/collection-card.component';
import { CollectionCarouselComponent } from './collection-carousel/collection-carousel.component';
import { SearchBarComponent } from './search-bar/search-bar.component';
import { GenerateTokenComponent } from './generate-token/generate-token.component';
import { LandingComponent } from './landing/landing.component';
import { AboutPageComponent } from './about-page/about-page.component';
import { NavigationComponent } from './navigation/navigation.component';
import { GenerateInputComponent } from './generate-input/generate-input.component';
import { generateReducer } from './generate-input/store/generate.reducer';
import {NewsEffects} from './generate-input/store/generate.effects';

@NgModule({
  declarations: [
    AppComponent,
    CollectionCardComponent,
    CollectionCarouselComponent,
    SearchBarComponent,
    GenerateTokenComponent,
    LandingComponent,
    AboutPageComponent,
    NavigationComponent,
    GenerateInputComponent
  ],
  imports: [
    FormsModule,
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    StoreModule.forRoot({generator: generateReducer}),
    EffectsModule.forRoot([NewsEffects]),
    BrowserAnimationsModule,
    MatProgressBarModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    NgxChartsModule,
    MarkdownModule.forRoot({ loader: HttpClient }),
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
