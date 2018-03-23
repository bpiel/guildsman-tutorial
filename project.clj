(defproject guildsman-tutorial "0.1.0-SNAPSHOT"
  :description "Tutorial for Guildsman, a TensorFlow libary for Clojure"
  :url "http://github.com/bpiel/guildsman-tutorial"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [com.billpiel/guildsman "0.0.1-DEV"]]
  :main ^:skip-aot guildsman-tutorial.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
