(ns guildsman-tutorial.core
  (:require [com.billpiel.guildsman.core :as gm]
            [com.billpiel.guildsman.ops.basic :as gb]
            [com.billpiel.guildsman.ops.composite :as gc]
            [com.billpiel.guildsman.dev :as gd]
            clojure.repl)
  (:gen-class))

;; Note the required namespaces...

(gb/add 1. 2.)

#_
;; =>
{:op :Add,
 :inputs [1.0 2.0],
 :ctrl-inputs nil,
 :id nil,
 :attrs {},
 :scope []}

;; Check out the docs!
;; They're auto-generated from TensorFlow, so sometimes they refer to
;; python concepts or docs.

(clojure.repl/doc gb/add)


;; These are all equivalent.

(gb/add :add1 1. 2.)
(gb/add {:id :add1} 1. 2.)
(gb/add :add1 {} 1. 2.)

#_
;; =>
{:op :Add,
 :inputs [1.0 2.0],
 :ctrl-inputs nil,
 :id :add1,
 :attrs {},
 :scope []}

(gm/set-global-tensor-conversion-scope!)

(gm/produce (gb/add 1. 2.)) ;; => 3.0

(gm/produce (gb/add [1. 2.] [3. 4.])) ;; => [4.0 6.0]

;; BUILD, RUN, EXE, FETCH, PRODUCE...

(def plan1 (gb/add 1. 2.))

(def graph1 (gm/build->graph plan1))

(def sess1 (gm/graph->session graph1))

(gm/fetch sess1 plan1) ;; => 3.0

(def plan2 (gb/mul plan1 1.5))

(gm/build->graph graph1 plan2) ;; Building is idempotent! Don't worry about duplicate nodes.

(gm/fetch sess1 plan2) ;; => 4.5

(def v1 (gc/vari :v1 [1. 3.]))

(gm/build->graph graph1 v1)

;; Get ready for an EXCEPTION!

(gm/fetch sess1 v1)

 ;; Attempting to use uninitialized value v1/variable [[Node: v1/read =
 ;;   Identity[T=DT_FLOAT,
 ;;   _device="/job:localhost/replica:0/task:0/device:CPU:0"](v1/variable)]]

(gm/run-global-vars-init sess1)

(gm/fetch sess1 v1) ;; => [1.0 3.0]

(def av1 (gb/assign :av1 v1 [1.2 3.1]))

(gm/build->graph graph1 av1)

(gm/run sess1 av1)

(gm/fetch sess1 v1) ;; => [1.2 3.1]

(gm/run-global-vars-init sess1)

(gm/fetch sess1 v1) ;; => [1.0 3.0]

(let [ph1 (gb/placeholder :ph1 gm/dt-float [2])
      add1 (gb/add 1. ph1)]

  ;; NOTE; Because `ph1` is an input to `add1`, building `add1` builds
  ;; `ph1` first. The opposite is not true, though.
  (gm/build->graph graph1 add1)

  ;; 1 + 1.3 =
  (gm/fetch sess1 add1 {:ph1 1.3})) ;; => 2.3

(let [v2 (gc/vari :v2 0.)
      inc-v2 (->> v2
                  (gb/add 1.)
                  (gb/assign v2))]
  (gm/build->graph graph1 inc-v2)
  (gm/run-global-vars-init sess1)
  (gm/run-all sess1 (repeat 100 inc-v2))
  (gm/fetch sess1 v2))
;; => 100.0

(gm/close sess1)

(gm/close graph1)

;; LOSS

(gm/produce
 (gc/mean-squared-error [0. 0.] [1. 2.])) ;; => 5.0

(gm/produce
 (gc/mean-squared-error [0.5 1.] [1. 2.])) ;; => 1.25

(gm/produce
 (gc/mean-squared-error [1. 1.5] [1. 2.])) ;; => 0.25

(gm/produce
 (gc/mean-squared-error [1. 2.] [1. 2.])) ;; => 0.0


(let [v (gc/vari :v [0. 0. 0.])
      goal [1. 2. 3.]
      loss (gc/mean-squared-error v goal)
      opt (gc/grad-desc-opt :opt 0.1 loss)
      sess (gm/build->session opt)
      _ (gm/run-global-vars-init sess)
      result (gm/fetch-all sess [v loss])]
  (gm/close sess)
  (gm/close (:graph sess))
  result)
;; ==> ([0. 0. 0.] 14.)

(let [v (gc/vari :v [0. 0. 0.])
      goal [1. 2. 3.]
      loss (gc/mean-squared-error v goal)
      opt (gc/grad-desc-opt :opt 0.1 loss)
      sess (gm/build->session opt)
      _ (gm/run-global-vars-init sess)
      _ (gm/run-all sess (repeat 10 opt)) ;; <============
      result (gm/fetch-all sess [v loss])]
  (gm/close sess)
  (gm/close (:graph sess))
  result)
;; => ([0.8926258 1.7852516 2.6778774] 0.16140904)

(let [v (gc/vari :v [0. 0. 0.])
      goal [1. 2. 3.]
      loss (gc/mean-squared-error v goal)
      opt (gc/grad-desc-opt :opt 0.1 loss)
      sess (gm/build->session opt)
      _ (gm/run-global-vars-init sess)
      _ (gm/run-all sess (repeat 100 opt))
      ;;                         ^^^
      result (gm/fetch-all sess [v loss])]
  (gm/close sess)
  (gm/close (:graph sess))
  result)
;; => ([0.9999999 1.9999998 2.9999995] 2.9842795E-13)

(defn train
  [n actual goal]
  (let [loss (gc/mean-squared-error actual goal)
        opt (gc/grad-desc-opt :opt 0.03 loss)
        sess (gm/build->session opt)
        _ (gm/run-global-vars-init sess)
        _ (gm/run-all sess (repeat n opt))
        result (gm/fetch-all sess [actual loss])]
    (gm/close sess)
    (gm/close (:graph sess))
    result))

(train 100
       (gc/vari :v [0. 0. 0.])
       [1. 2. 3.])

(let [inputs [[0. 0.]
              [0. 1.]
              [1. 0.]
              [1. 1.]]
      goal [[0.] [1.] [1.] [0.]]
      actual (gc/dense {:units 1} inputs)]
  (train 10
         actual
         goal))

(let [inputs [[0. 0.]
              [0. 1.]
              [1. 0.]
              [1. 1.]]
      goal [[0.] [1.] [1.] [0.]]
      actual (gc/dense {:units 1} inputs)]
  (train 100
         actual
         goal))

(let [inputs [[0. 0.]
              [0. 1.]
              [1. 0.]
              [1. 1.]]
      goal [[0.] [1.] [1.] [0.]]
      actual (->> inputs
                  (gc/dense {:units 2})
                  (gc/dense {:units 1}))]
  (train 1000
         actual
         goal))

(let [inputs [[0. 0.]
              [0. 1.]
              [1. 0.]
              [1. 1.]]
      goal [[0.] [1.] [1.] [0.]]
      actual (->> inputs
                  (gc/dense {:units 2
                             :activation gb/tanh})
                  (gc/dense {:units 1}))]
  (train 1000
         actual
         goal))

;; ========= BASIC MNIST

#_(gm/pkg-dl-repo! "https://bpiel.github.io/guildsman-packages/pkgs.edn")

(gm/pkg-dl-repo! "http://localhost:8000/pkgs.edn")

#_(clojure.pprint/pprint @com.billpiel.guildsman.packages/registry)

(gm/def-workspace mnist1
  ;; TODO let$
  (gm/let+ [{:keys [features labels socket]}
            (->> (gc/dsi-socket :socket
                                {:fields [:features gm/dt-float [-1 784]
                                          :labels   gm/dt-int   [-1]]})
                 (gc/dsi-socket-outputs))

            {:keys [logits classes]}
            (+->> features
                  (gc/dense {:activation gb/relu :units 1024})
                  (gc/dense {:id :logits :units 10})
                  (gb/arg-max :classes $ 1))

            {:keys [loss opt]}
            (+->> labels
                  (gc/one-hot $ 10)
                  (gc/mean-squared-error :loss logits)
                  (gc/grad-desc-opt :opt 0.05))

            acc (gc/accuracy :acc
                             (gc/cast-tf gm/dt-int
                                         classes)
                             labels)]
    
    {:plans [acc opt]
     :modes {:train {:step [opt]
                     ::gd/summaries [acc loss logits classes]
                     :fetch [acc]
                     :iters {socket (gc/dsi-plug {:batch-size 300
                                                  :epoch-size 300}
                                                 [:bpiel/mnist-train-60k-labels-v1
                                                  :bpiel/mnist-train-60k-features-v1])}}
             :test {::gd/summaries [acc loss]
                    :fetch [acc]
                    :iters {socket (gc/dsi-plug {:batch-size -1 ;; TODO removable?
                                                 :epoch-size 1000} 
                                                [:bpiel/mnist-test-10k-features-v1
                                                 :bpiel/mnist-test-10k-labels-v1])}}
             :predict {:feed-args [features]
                       :fetch-return [classes]}}
     :repo {:path "/tmp/gm-repo1"}}))

(gm/pkg-set-repo-path! "/tmp/gm-pkgs")

(gm/pkg-prefetch-all-assets-sync mnist1)

(def train-test
  (gm/mk-train-test-wf
   {:plugins [gd/plugin gm/gm-plugin]
    :duration [:steps 1000]
    :interval [:steps 100]
    :chkpt-interval [:secs 3600]}))

(gm/start-wf train-test mnist1)

(gm/ws-pr-status mnist1)





;; THE GOAL? ============================================


(gm/def-workspace mnist2
  ;; TODO let$
  (gm/let+ [{:keys [features labels socket]}
            (->> (gc/dsi-socket :socket
                                {:fields [:features gm/dt-float [-1 784]
                                          :labels   gm/dt-int   [-1]]})
                 (gc/dsi-socket-outputs))

            {:keys [logits classes conv1 conv2]}
            (+->> features
                  #_(gb/sub :data2 $ mnist/img-mean)
                  (gb/reshape $ [-1 28 28 1])
                  (gc/conv2d {:id :conv1 :filters 32 :kernel-size [5 5]
                              :padding "SAME" :activation gb/relu})
                  (gc/max-pooling2d {:pool-size [2 2] :strides [2 2]})
                  (gc/conv2d {:id :conv2 :filters 64 :kernel-size [5 5]
                              :padding "SAME" :activation gb/relu})
                  (gc/max-pooling2d {:pool-size [2 2]
                                     :strides [2 2]})
                  (gb/reshape $ [-1 (* 4 784)])
                  (gc/dense {:activation gb/relu :units 1024})
                  (gc/dense {:id :logits :units 10})
                  (gb/arg-max :classes $ 1))

            {:keys [loss opt]}
            (+->> labels
                  (gc/one-hot $ 10)
                  (gc/mean-squared-error :loss logits)
                  (gc/grad-desc-opt :opt 0.05))

            acc (gc/accuracy :acc
                             (gc/cast-tf gm/dt-int
                                         classes)
                             labels)]
    
    {:plans [acc opt]
     :modes {:train {:step [opt]
                     ::gd/summaries [acc loss logits classes conv1 conv2]
                     :fetch [acc]
                     :iters {socket (gc/dsi-plug {:batch-size 300
                                                  :epoch-size 300}
                                                 [:bpiel/mnist-train-60k-labels-v1
                                                  :bpiel/mnist-train-60k-features-v1])}}
             :test {::gd/summaries [acc loss]
                    :fetch [acc]
                    :iters {socket (gc/dsi-plug {:batch-size -1 ;; TODO removable?
                                                 :epoch-size 1000} 
                                                [:bpiel/mnist-test-10k-features-v1
                                                 :bpiel/mnist-test-10k-labels-v1])}}
             :predict {:feed-args [features]
                       :fetch-return [classes]}}
     :repo {:path "/tmp/gm-repo1"}}))

#_(gm/pkg-dl-repo! "https://bpiel.github.io/guildsman-tutorials/packages.edn")

#_(gm/pkg-prefetch-all mnist1)

(def train-test
  (gm/mk-train-test-wf
   {:plugins [gd/plugin gm/gm-plugin]
    :duration [:steps 1000]
    :interval [:steps 100]
    :chkpt-interval [:secs 3600]}))

(gm/start-wf train-test mnist1)

;; see fetched acc
;; web app
;; inspect checkpoints, select one
;; run predict

(gm/ws-pr-status mnist1)

(def predict
  (gm/mk-predict-wf
   {:plugins [gd/plugin gm/gm-plugin]}))

;; =================================================


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, Guilsman!"))
