# Gaussian Processes Practical Tips

These repo contains a collection of experiments on benchmark datasets to validate the practical tips to improve the performance of Gaussian processes.

## Tips

* [A practical guide to Gaussian processes](https://infallible-thompson-49de36.netlify.app/)

## Datasets

* [UCI datasets](https://drive.google.com/u/0/uc?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU) (available on request)

## Collaboration guidelines
### General guidelines
* You can take up a particular issue, discuss doubts in the issue itself and submit a PR to resolve the issue.

### Technical details
* We are going to work with `hydra` for configuration management. For motivation behind using it, I highly recommend watching [this video](https://youtu.be/tEsPyYnzt8s).
* If needed, we should be able to spin off multiple jobs (experiment configs) at once using hydra's cli interface.
* We should log statistics such as training and testing time, memory usage etc. using `hydra`'s logging interface.