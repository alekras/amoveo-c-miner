-module(miner).

-export([start/0]).
-define(Peer, "http://159.65.120.84:8085"). %for a mining pool on the server.
-define(CORES, 2).
-define(Pubkey, <<"BGv90RwK8L4OBSbl+6SUuyWSQVdkVDIOJY0i1wpWZINMTIBAM9/z3bOejY/LXm2AtA/Ibx4C7eeTJ+q0vhU9xlA=">>). %% 88 bytes 704 bits

-ifdef(MACOS).
  -define(Treshold, 7).%how long to wait in seconds before checking if new mining data is available.
  -define(TracePeriod_1, 1).%how long to wait in seconds while checking for success in GPU.
  -define(TracePeriod_2, 1).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(HTTPC, httpc_mock).
-else.
  -define(Treshold, 56).%how long to wait in seconds before checking if new mining data is available.
  -define(TracePeriod_1, 2).%how long to wait in seconds while checking for success in GPU.
  -define(TracePeriod_2, 1).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1000).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(HTTPC, httpc).
-endif.

-define(PORT_NAME, amoveo_c_miner).
-define(USE_SHARE_POOL, true).

start() ->
  lager:start(),
  lager:info("  Let start mining.", []),
  os:cmd("pkill " ++ atom_to_list(?PORT_NAME)),
  timer:sleep(1000),
  Ports = start_many(?CORES),
  start_c_miners(Ports).

ask_for_work() ->
  Data = 
  case ?USE_SHARE_POOL of
    true -> <<"[\"mining_data\",\"", ?Pubkey/binary, "\"]">>;
    false -> <<"[\"mining_data\"]">>
  end,
  R = talk_helper(Data, ?Peer, 1000),
  try
    [<<"ok">>, [_, Hash, BlockDiff, ShareDiff]] = mochijson2:decode(R),
    F = base64:decode(Hash),
    case ?USE_SHARE_POOL of
      true -> {F, BlockDiff, ShareDiff};
      false -> {F, ShareDiff, ShareDiff}
    end
  catch _:_ ->
    lager:debug("Server wrong response : ~128p~n", [R]),
    timer:sleep(1000),
    ask_for_work()
  end.

flush() ->
  receive
    _ -> flush()
  after 0 -> 
    ok
  end.

start_c_miners(Ports) ->
  {F, BD, SD} = ask_for_work(),
  lager:debug(" ASK for work: Hash:~256p Diff:~p / ~p~n", [F, BD, SD]),
  start_miner_step(Ports, F, BD, SD, ?Treshold),
  start().

start_miner_step(Ports, F, BD, SD, Period) ->
  RS = crypto:strong_rand_bytes(23),
  flush(),
  [Port ! {self(), {command, <<"I", F/binary, RS/binary, BD:32/integer, SD:32/integer, Core_id:32/integer>>}} || {Port, Core_id} <- Ports],
  wait_for(Ports, <<"I">>, ?CORES),
  run_miners(Ports, F, Period).

wait_for(_, _, 0) -> pass;
wait_for(Ports, Command, N) ->
  receive
    {_Port, {data, Command}} ->
      wait_for(Ports, Command, N - 1);
    {_Port, {data, <<1:8, Nonce:23/binary, _Hash:32/binary>>}} ->
%        io:format("Success:~p, Nonce:~128p~n Hash:~128p~n", [Success, Nonce, Hash]),
      BinNonce = base64:encode(Nonce),
      Data = <<"[\"work\",\"", BinNonce/binary, "\",\"", ?Pubkey/binary, "\"]">>,
      RR = talk_helper(Data, ?Peer, 2),
      LL = 
        try mochijson2:decode(RR) of
          [_ | L] -> L;
          L -> L
        catch E:Reason ->
          lager:debug(" ---- Unexpected response from server: ~128p~nError: ~p : ~p~n", [RR, E, Reason]),
          RR
        end,
      PortIdx = proplists:get_value(_Port, Ports, -1),
      lager:warning("Port[~p] founds a nonce ~128p. Response from server: ~128p~n", [PortIdx, BinNonce, LL]),
%%      check_data(_Hash, Nonce),
      wait_for(Ports, Command, N - 1);
    Err -> 
      lager:debug(" Command: ~p. Err from port: ~p", [Command, Err]),
      wait_for(Ports, Command, N - 1)
  after 1000 ->
      lager:debug(" port timeout after command: ~p~n", [Command]),
      wait_for(Ports, Command, N - 1)
  end.

run_miners(Ports, Bhash, Period) ->
  flush(),
  [Port ! {self(), {command, <<"U">>}} || {Port, _Core_id} <- Ports],
  wait_for(Ports, <<"U">>, ?CORES),

  if Period >= ?Treshold ->
    timer:sleep(?TracePeriod_2 * 1000),
    {F, BD, SD} = ask_for_work(),
    lager:debug("#~2.2.0w ask for work: Hash:~256p Diff:~p / ~p~n", [Period, F, BD, SD]),
    if F =:= Bhash ->
      run_miners(Ports, Bhash, Period + ?TracePeriod_2);
    true ->
      flush(),
      [Port ! {self(), {command, <<"S">>}} || {Port, _Core_id} <- Ports],
      wait_for(Ports, <<"S">>, ?CORES),
      if Period =:= ?Treshold ->
        start_miner_step(Ports, F, BD, SD, 10);
      true ->
        start_miner_step(Ports, F, BD, SD, 0)
      end
    end;
  true ->
    timer:sleep(?TracePeriod_1 * 1000),
    run_miners(Ports, Bhash, Period + ?TracePeriod_1)
  end.

start_many(N) when N < 1 -> [];
start_many(N) -> 
  Opts = [{packet, 4}, binary, exit_status, use_stdio, {args, [integer_to_list(N - 1)]}],
  Port = erlang:open_port({spawn_executable, ?PORT_NAME}, Opts),
  [{Port, N - 1} | start_many(N - 1)].

talk_helper2(Data, Peer) ->
  ?HTTPC:request(post, {Peer, [], "application/octet-stream", iolist_to_binary(Data)}, [{timeout, 3000}], []).

talk_helper(_Data, _Peer, 0) -> []; %%throw("talk helper failed");
talk_helper(Data, Peer, N) ->
  case talk_helper2(Data, Peer) of
    {ok, {_Status, _Headers, []}} ->
      lager:debug("server ~p gave confusing response: ~p; ~p.\n", [Peer, _Status, _Headers]),
      timer:sleep(?Pool_sleep_period),
      talk_helper(Data, Peer, N-1);
    {ok, {_, _, R}} ->
      R;
    _E -> 
      lager:debug("\nIf you are running a solo-mining node, then this error may have happened because you need to turn on and sync your Amoveo node before you can mine. You can get it here: https://github.com/zack-bitcoin/amoveo \n If this error happens while connected to the public mining node, then it can probably be safely ignored."),
      timer:sleep(?Pool_sleep_period),
      talk_helper(Data, Peer, N-1)
  end.

check_data(Bhash, Nonce) ->
  H = hash:doit(<<Bhash/binary, Nonce/binary>>),
  I = pow:hash2integer(H, 1),
  J = pow:hash2integer(H, 0),
  lager:warning(" check data: ~256p Diff: ~p / ~p~n", [H, I, J]).

%% datetime_string() ->
%%   {{_Year, Month, Day}, {Hour, Minute, Second}} = calendar:local_time(),
%%   lists:flatten(io_lib:format(" ~2.2.0w/~2.2.0w-~2.2.0w:~2.2.0w:~2.2.0w ", [Month, Day, Hour, Minute, Second])).

