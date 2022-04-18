import { enableProdMode } from '@angular/core';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { AppModule } from './app/app.module';
import { environment } from './environments/environment';
import { initializeApp } from "firebase/app";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAsSi6U5s75w86gjcx0aVfW2Wei5mZ0XZs",
  authDomain: "nf2-web.firebaseapp.com",
  projectId: "nf2-web",
  storageBucket: "nf2-web.appspot.com",
  messagingSenderId: "172097390437",
  appId: "1:172097390437:web:792fc4e71a811bba36438c"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

if (environment.production) {
  enableProdMode();
}

platformBrowserDynamic().bootstrapModule(AppModule)
  .catch(err => console.error(err));
