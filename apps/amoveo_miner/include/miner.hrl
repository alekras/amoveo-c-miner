-record(state, {
  ports::list(),
  given_hash::binary(),
  difficulty::integer(),
  command::list(),
  msg_count::integer(),
  period::integer()
}).

-define(Peer, "http://159.65.120.84:8085").
-define(Pubkey, <<"BGv90RwK8L4OBSbl+6SUuyWSQVdkVDIOJY0i1wpWZINMTIBAM9/z3bOejY/LXm2AtA/Ibx4C7eeTJ+q0vhU9xlA=">>). %% 88 bytes 704 bits
-define(PORT_NAME, amoveo_c_miner).
-define(USE_SHARE_POOL, true).
-define(CORES, 2).
-ifdef(MACOS).
  -define(Treshold, 7).%how long to wait in seconds before checking if new mining data is available.
  -define(Treshold_1, 2).
  -define(TracePeriod_1, 1).%how long to wait in seconds while checking for success in GPU.
  -define(TracePeriod_2, 1).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(HTTPC, httpc_mock).
-else.
  -define(Treshold, 56).%how long to wait in seconds before checking if new mining data is available.
  -define(Treshold_1, 10).
  -define(TracePeriod_1, 2).%how long to wait in seconds while checking for success in GPU.
  -define(TracePeriod_2, 1).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1000).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(HTTPC, httpc).
-endif.
