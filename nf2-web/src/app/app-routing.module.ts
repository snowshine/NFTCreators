import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {LandingComponent} from './landing/landing.component';
import {GenerateTokenComponent} from './generate-token/generate-token.component';
import {GenerateInputComponent} from './generate-input/generate-input.component';
import { AboutPageComponent } from './about-page/about-page.component';

const routes: Routes = [
  { path: 'generate', component: GenerateTokenComponent },
  { path: 'generate/:slug', component: GenerateInputComponent },
  { path: 'blog', component: AboutPageComponent },
  { path: '**', component: LandingComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
