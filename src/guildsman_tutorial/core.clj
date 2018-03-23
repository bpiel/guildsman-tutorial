(ns guildsman-tutorial.core
  (:require [com.billpiel.guildsman.core :as gm]
            [com.billpiel.guildsman.ops.basic :as gb])
  (:gen-class))

(com.billpiel.guildsman.tensor-scope/set-global-conversion-scope!)

(gm/produce (gb/add 1. 2.))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, Guilsman!"))


















































