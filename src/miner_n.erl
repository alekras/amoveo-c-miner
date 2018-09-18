-module(miner_n).

-export([start/0, send_comm_2_port/4]).
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
  -define(Treshold, 54).%how long to wait in seconds before checking if new mining data is available.
  -define(TracePeriod_1, 2).%how long to wait in seconds while checking for success in GPU.
  -define(TracePeriod_2, 1).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1000).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(HTTPC, httpc).
-endif.

-define(PORT_NAME, amoveo_c_miner).
-define(USE_SHARE_POOL, true).

start() ->
  io:format("~n~s Started mining.~n~n", [datetime_string()]),
  os:cmd("pkill " ++ atom_to_list(?PORT_NAME)),
  timer:sleep(1000),
  Ports = start_many(?CORES),
  start_c_miners(Ports).

ask_for_work() ->
  Data = case ?USE_SHARE_POOL of
    true -> <<"[\"mining_data\",\"", ?Pubkey/binary, "\"]">>;
    false -> <<"[\"mining_data\"]">>
	end,
  R = talk_helper(Data, ?Peer, 1000),
  if is_list(R) ->
    unpack_mining_data(R);
  true ->
    io:format("Server wrong response : ~128p~n", [R]),
    timer:sleep(1000),
    ask_for_work()
  end.
  
flush() ->
  receive
    _ -> flush()
  after 0 -> 
    ok
  end.

unpack_mining_data(R) ->
  [<<"ok">>, [_, Hash, BlockDiff, ShareDiff]] = mochijson2:decode(R),
  F = base64:decode(Hash),
  case ?USE_SHARE_POOL of
    true ->
      {F, BlockDiff, ShareDiff};
    false ->
      {F, ShareDiff, ShareDiff}
  end.

start_c_miners(Ports) ->
  {F, BD, SD} = ask_for_work(),
  io:format("~s ASK for work: Hash:~256p Diff:~p / ~p~n", [datetime_string(), F, BD, SD]),
  start_miner_step(Ports, F, BD, SD, ?Treshold),
%%  [Port ! {self(), close} || {_, Port} <- Ports],
  start().

start_miner_step(Ports, F, BD, SD, Period) ->
  RS = crypto:strong_rand_bytes(23),
  flush(),
  M_pid = self(),
  [erlang:spawn(?MODULE, send_comm_2_port, [Port, Core_id, <<"I", F/binary, RS/binary, BD:32/integer, SD:32/integer, Core_id:32/integer>>, M_pid]) || {Core_id, Port} <- Ports],
  wait_for(?CORES),
  run_miners(Ports, F, Period).

send_comm_2_port(Port, Dev_id, <<"I", _, _, _, _, _>> = Msg, M_pid) ->
  Port ! {self(), {command, Msg}},
  receive
    {_Port, {data, <<"I">>}} ->
      M_pid ! {ok, Dev_id};
    Err -> 
      M_pid ! {port_error, Dev_id, Err}
  after 1000 ->
      M_pid ! {port_timeout, Dev_id}
  end;
send_comm_2_port(Port, Dev_id, <<"U">> = Msg, M_pid) ->
  Port ! {self(), {command, Msg}},
  receive
    {_Port, {data, <<"U">>}} ->
      M_pid ! {ok, Dev_id};
    {_Port, {data, <<Success:8, Nonce:23/binary, _Hash:32/binary>>}} ->
%     io:format("Success:~p, Nonce:~128p~n", [Success, Nonce]),
      if Success == 1 ->
%        io:format("Success:~p, Nonce:~128p~n Hash:~128p~n", [Success, Nonce, Hash]),
        BinNonce = base64:encode(Nonce),
        Data = <<"[\"work\",\"", BinNonce/binary, "\",\"", ?Pubkey/binary, "\"]">>,
        RR = talk_helper(Data, ?Peer, 2),
        LL = 
        try mochijson2:decode(RR) of
          [_ | L] -> L;
          L -> L
        catch E:Reason ->
          io:format(" ---- Unexpected response from server: ~128p~nError: ~p : ~p~n", [RR, E, Reason]),
          RR
        end,
        M_pid ! {success, Dev_id},
        io:format("~s {~p} !!!!! Found a block. Nonce ~128p. Response from server: ~128p~n", [datetime_string(), Dev_id, BinNonce, LL]),
        check_data(_Hash, Nonce);
%%            {F1, BD1, SD1} = ask_for_work(),
%%            io:format("~s/~2.2.0w Ask for work: Hash:~256p Diff:~p / ~p~n", [datetime_string(), Period, F1, BD1, SD1]),
%%            start_miner_step(Ports, F1, BD1, SD1, ?Treshold);
      true ->
        M_pid ! {ok, Dev_id} %% ?? maybe fail ?
      end;
    Err -> 
      M_pid ! {port_error, Dev_id, Err}
  after 1000 ->
      M_pid ! {port_timeout, Dev_id}
  end;
send_comm_2_port(Port, Dev_id, <<"S">> = Msg, M_pid) ->
  Port ! {self(), {command, Msg}},
  receive
    {_Port, {data, <<"S">>}} ->
      M_pid ! {ok, Dev_id};
    Err -> 
      M_pid ! {port_error, Dev_id, Err}
  after 1000 ->
      M_pid ! {port_timeout, Dev_id}
  end.

wait_for(0) -> pass;
wait_for(N) ->
  receive
    {ok, _Dev_id} ->
      wait_for(N - 1);
    {port_error, Dev_id, Err} ->
      io:format(" ~s Err from port[~p]: ~p", [datetime_string(), Dev_id, Err]),
      wait_for(N - 1);
    {port_timeout, Dev_id} ->
      io:format(" ~s port[~p] timeout error: ~n", [datetime_string(), Dev_id]),
      wait_for(N - 1)
  after 1500 ->
      wait_for(N - 1)
  end.

run_miners(Ports, Bhash, Period) ->
  flush(),
  M_pid = self(),
  [erlang:spawn(?MODULE, send_comm_2_port, [Port, Core_id, <<"U">>, M_pid]) || {Core_id, Port} <- Ports],
  wait_for(?CORES),

  if Period >= ?Treshold ->
    timer:sleep(?TracePeriod_2 * 1000),
    {F, BD, SD} = ask_for_work(),
    io:format("~s#~2.2.0w ask for work: Hash:~256p Diff:~p / ~p~n", [datetime_string(), Period, F, BD, SD]),
    if F =:= Bhash ->
      run_miners(Ports, Bhash, Period + ?TracePeriod_2);
    true ->
      flush(),
      [erlang:spawn(?MODULE, send_comm_2_port, [Port, Core_id, <<"S">>, M_pid]) || {Core_id, Port} <- Ports],
      wait_for(?CORES),
      start_miner_step(Ports, F, BD, SD, 0)
    end;
  true ->
    timer:sleep(?TracePeriod_1 * 1000),
    run_miners(Ports, Bhash, Period + ?TracePeriod_1)
  end.

start_many(N) when N < 1 -> [];
start_many(N) -> 
  Opts = [{packet, 4}, binary, exit_status, use_stdio, {args, [integer_to_list(N - 1)]}],
  Port = erlang:open_port({spawn_executable, ?PORT_NAME}, Opts),
  [{N - 1, Port} | start_many(N - 1)].

talk_helper2(Data, Peer) ->
  ?HTTPC:request(post, {Peer, [], "application/octet-stream", iolist_to_binary(Data)}, [{timeout, 3000}], []).

talk_helper(_Data, _Peer, 0) -> []; %%throw("talk helper failed");
talk_helper(Data, Peer, N) ->
  case talk_helper2(Data, Peer) of
    {ok, {_Status, _Headers, []}} ->
      io:fwrite("server ~p gave confusing response: ~p; ~p.\n", [Peer, _Status, _Headers]),
      timer:sleep(?Pool_sleep_period),
      talk_helper(Data, Peer, N-1);
    {ok, {_, _, R}} ->
      R;
    _E -> 
      io:fwrite("\nIf you are running a solo-mining node, then this error may have happened because you need to turn on and sync your Amoveo node before you can mine. You can get it here: https://github.com/zack-bitcoin/amoveo \n If this error happens while connected to the public mining node, then it can probably be safely ignored."),
      timer:sleep(?Pool_sleep_period),
      talk_helper(Data, Peer, N-1)
  end.

check_data(Bhash, Nonce) ->
  H = hash:doit(<<Bhash/binary, Nonce/binary>>),
  I = pow:hash2integer(H, 1),
  J = pow:hash2integer(H, 0),
  io:format("~s check data: ~256p Diff: ~p / ~p~n", [datetime_string(), H, I, J]).

  
datetime_string() ->
  {{_Year, Month, Day}, {Hour, Minute, Second}} = calendar:local_time(),
  lists:flatten(io_lib:format("~2.2.0w/~2.2.0w-~2.2.0w:~2.2.0w:~2.2.0w", [Month, Day, Hour, Minute, Second])).

