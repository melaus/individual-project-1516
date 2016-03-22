# Notes

## Predictions
- speed of prediction is very slow:
  - more than 2 hours to predict 290K patches on Balena (1 node) [svc_rbf_6, 3 trained classes]


## Data Type 
- numpy arrays are much more efficient - less space, hence less time building and loading them
    - 308 VS 108 seconds (per_pixel)


## Challenges
- data engineering requires very careful consideration
    - spent a lot of time refining the datset and making it into an easy-to-manipulate format
    - vision VS reality - when it comes to fitting a model
- fitting model takes a large amount of time
    - cannot simply run many models in a short amount of time
    - debbuging is difficult - requires in-depth exploration to understand what has went wrong
    - the problem of generalisation - want a model that is efficient and predictive

# Cross Validation
- cv = cv
    - read up and see what options we can choose, or whether to just choose a random number
