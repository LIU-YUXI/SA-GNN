# SA-GNN

## Environment

- python=3.6.12

- tensorflow-gpu=1.14.0

- numpy=1.16.0

- scipy=1.5.2

## Datasets

We utilized three datasets to evaluate: *Gowalla, MovieLens,Yelp* and *Amazon*. Following the common settings of implicit feedback, if user  has rated item , then the element  is set as 1, otherwise 0. We filtered out users and items with too few interactions.

We employ the most recent interaction as the test set, the penultimate interaction as the validation set, and the remaining interactions in the user behavior sequence as thetraining data.

## How to Run the Code

you need to create the `History/` and the `Models/` directories. The command to train SLSG on the Gowalla/MovieLens/Amazon dataset is as follows.

- Gowalla

```
./gowalla.sh > ./gowalla.log 
```

- MovieLens

```
./movielens.sh > ./movielens.log 
```

- Amazon

```
./amazon.sh > ./amazon.log 
```

- Yelp

```
./yelp.sh > ./yelp.log 
```