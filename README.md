# otto-recommender
Recommender engine based on the otto dataset


## Notebooks

There are 3 notebooks of interest:

* [architecture.ipynb](./architecture.ipynb)
  * Architecture diagrams and short discussion points
* [prep-data.ipynb](./prep-data.ipynb)
  * Basically a script to read and normalize the `OTTO` dataset
* [train_model.ipynb](./train_model.ipynb)
  1. loads the normalized dataset
  2. explores and plots features from the dataset
  3. trains the `Word2Vec` model based on the filtered data
  4. compares the model against a random baseline algorithm with `recall@k` and `Mean Reciprocal Rank`


### Service

`service.py` contains a minimal `flask` service with 2 endpoints:

* `GET:api/v1/popular`, returns the 10 most popular `aid`s
* `GET:api/v1/recommendations/<aid>`, returns 10 recommendations based on a single `aid`

`GET:api/v1/recommendations/<aid>` responds on average within `6` to `8` milliseconds on the same host. Whether or not this is "realtime" would be up to the Service Level Agreement.

My take is: yes, this is absolutely real-time when considering how long a regular website takes to respond, render and display. Add to this the fact that this recommendation can be queried in parallel with whatever else the website needs at checkout time, I would feel very comfortable with this latency. 

I imagine the client would need to fetch the images and product info based on the `aid` after receiving this reponse, however, all product images and information should either be queried in batch or in parallel, meaning we only have to pay the price of `recommendation_latency + slowest_product_fetch_time` in terms of latency. 

### Scalability

While python doesn't support multithreading, you could scale up by launching multiple processes. This is, however, very expensive in terms of memory. 
We note that the workload we have is _cpu bound_, as it relies on linear algebra (matrix vector multiplication). Furthermore, in a shared-memory setup, we can share the _word vectors_ in memory, as the operation we have only _reads_ the vectors. Note that we cannot achieve this shared memory setup if we're splitting up into multiple python processes.

In the end, I'd suggest any language which supports some kind of [green threads](https://en.wikipedia.org/wiki/Green_thread) with shared memory and easy access to fast linear algebra operations: 

* Rust with [Tokio](https://tokio.rs/)
* JVM based language
  * Java with [virtual threads (since 21)](https://openjdk.org/jeps/444)
  * Kotlin with coroutines
  * Scala with [Cats Effect](https://typelevel.org/cats-effect/) or [Zio](https://zio.dev/)

> NOTE: wrt. jvm languages, special care would go into avoiding copying between the JVM and native memory when calling out to an efficient linear algebra library. [There are many approaches available](https://developer.okta.com/blog/2022/04/08/state-of-ffi-java)



Using a different language does involve a more careful approach in terms of serializing and deserializing the word-vectors, as well as a better understanding of how the `gensim Word2Vec` models are queried "underneath the hood".

# Summary

The `Word2Vec` model scores around `4%` in the `recall@20` score. The random model almost always score a flat `0%`. We observe that the vocabulary (number of unique product IDs) is very large, wheighing in at `657940` which presents a challenging scenario compared to smaller vocabularies. Furthermore, the distribution is very tail-heavy:

![](aid_counts.png)

That is, the vocabulary contains many rarely used words. It is not clear whether or not that presents a particular challenge to the `Word2Vec` approach compared to other approaches.