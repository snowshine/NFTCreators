import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-about-page',
  templateUrl: './about-page.component.html',
  styleUrls: ['./about-page.component.css']
})
export class AboutPageComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
    const documentRoot = document.getElementById('app-wrapper');
    if (documentRoot) {
      documentRoot.style.backgroundColor = 'white'
    }

    const navigationHeader = document.getElementById('navigation-header');
    if (navigationHeader) {
      navigationHeader.style.backgroundColor = '#07042D';
    }
  }

  ngOnDestroy(): void {
    const documentRoot = document.getElementById('app-wrapper');
    if (documentRoot) {
      documentRoot.style.backgroundColor = '#07042D'
    }

    const navigationHeader = document.getElementById('navigation-header');
    if (navigationHeader) {
      navigationHeader.style.backgroundColor = '';
    }
  }
}
