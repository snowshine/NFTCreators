import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-collection-card',
  templateUrl: './collection-card.component.html',
  styleUrls: ['./collection-card.component.css']
})
export class CollectionCardComponent implements OnInit {
  @Input() collectionName: string = '';

  ngOnInit(): void {
  }

}
