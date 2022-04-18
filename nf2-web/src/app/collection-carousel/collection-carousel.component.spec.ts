import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CollectionCarouselComponent } from './collection-carousel.component';

describe('CollectionCarouselComponent', () => {
  let component: CollectionCarouselComponent;
  let fixture: ComponentFixture<CollectionCarouselComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CollectionCarouselComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CollectionCarouselComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
