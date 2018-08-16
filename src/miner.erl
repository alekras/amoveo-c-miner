-module(miner)
.
-export([start/0]).
%-define(Peer, "http://localhost:3011/").%for a full node on same computer.
%-define(Peer, "http://localhost:8081/").%for a full node on same computer.
%-define(Peer, "http://amoveopool2.com/work").%for a mining pool on the server.
%-define(Peer, "http://localhost:8085/").%for a mining pool on the same computer.
-define(Peer, "http://159.65.120.84:8085").%for a mining pool on the server.
-define(CORES, 1).
-define(Pubkey, <<"BGv90RwK8L4OBSbl+6SUuyWSQVdkVDIOJY0i1wpWZINMTIBAM9/z3bOejY/LXm2AtA/Ibx4C7eeTJ+q0vhU9xlA=">>). %% 88 bytes 704 bits

-ifdef(MACOS).
  -define(Treshold, 7).%how long to wait in seconds before checking if new mining data is available.
  -define(TracePeriod, 1).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(Miner_sleep, 10). %This is how you reduce the load on CPU. It sleeps this long in miliseconds between mining cycles.
  -define(HTTPC, httpc_mock).
-else.
  -define(Treshold, 54).%how long to wait in seconds before checking if new mining data is available.
  -define(TracePeriod, 2).%how long to wait in seconds while checking for changing mining data.
  -define(Pool_sleep_period, 1000).%How long to wait in miliseconds if we cannot connect to the mining pool.
  -define(Miner_sleep, 1000). %This is how you reduce the load on CPU. It sleeps this long in miliseconds between mining cycles.
  -define(HTTPC, httpc).
-endif.

-define(PORT_NAME, amoveo_c_miner).
-define(USE_SHARE_POOL, true).

start() ->
  io:format("~n~s Started mining.~n~n", [datetime_string()]),
  os:cmd("pkill " ++ atom_to_list(?PORT_NAME)),
  timer:sleep(?Miner_sleep),
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
  after 0 -> ok
  end.

unpack_mining_data(R) ->
  [<<"ok">>, [_, Hash, BlockDiff, ShareDiff]] = mochijson2:decode(R),
  F = base64:decode(Hash),
  io:format("~s ask for work. Server responce: Hash:~256p Diff:~p / ~p~n", [datetime_string(), F, BlockDiff, ShareDiff]),
  case ?USE_SHARE_POOL of
	  true ->
      {F, BlockDiff, ShareDiff};
		false ->
      {F, ShareDiff, ShareDiff}
	end.

start_c_miners(Ports) ->
  {F, BD, SD} = ask_for_work(),
  run_miners(Ports, F, BD, SD, ?Treshold + 1),
%%  [Port ! {self(), close} || {_, Port} <- Ports],
  start().

run_miners(Ports, F, BD, SD, Period) ->
  RS = crypto:strong_rand_bytes(23),
  flush(),
  [Port ! {self(), {command, <<"I", F/binary, RS/binary, BD:32/integer, SD:32/integer, Core_id:32/integer>>}} || {Core_id, Port} <- Ports],
%  io:format("~s Sent command 'I'~n", [datetime_string()]),
  run_miners(Ports, F, Period).

run_miners(Ports, Bhash, Period) ->
  flush(),
  [Port ! {self(), {command, <<"U">>}} || {_Core_id, Port} <- Ports],
%  io:format("~s Sent command 'U'~n", [datetime_string()]),
  receive
    {_Port, {data, <<"U">>}} ->
      ok;
    {_Port, {data, <<Success:8, Nonce:23/binary, Hash:32/binary>>}} ->
     io:format("Success:~p, Nonce:~128p~n", [Success, Nonce]),
      if Success == 1 ->
           io:format("Success:~p, Nonce:~128p~n Hash:~128p~n", [Success, Nonce, Hash]),
           BinNonce = base64:encode(Nonce),
           Data = <<"[\"work\",\"", BinNonce/binary, "\",\"", ?Pubkey/binary, "\"]">>,
           check_data(Bhash, Nonce),
           io:format("Data to server: ~128p~n", [Data]),
           RR = talk_helper(Data, ?Peer, 2),
           LL = 
           try mochijson2:decode(RR) of
             [_ | L] -> L;
             L -> L
           catch E:Reason ->
             io:format(" ---- Unexpected response from server: ~128p~nError: ~p : ~p~n", [RR, E, Reason]),
             RR
           end,
           io:format("~s Found a block. Nonce ~128p. Response from server: ~128p~n~n", [datetime_string(), BinNonce, LL]),
           {F1, BD1, SD1} = ask_for_work(),
           run_miners(Ports, F1, BD1, SD1, ?Treshold + 1);
         true ->
           ok %%io:format("~s Did not find a block in that period~n", [datetime_string()])
      end;
    Err -> 
      io:format("U) Err from port: ~p", [Err]),
      start()
  after 10000 ->
      io:format("~s U) PORT timeout error: ~n", [datetime_string()]),
      start()
  end,

  timer:sleep(?TracePeriod * 1000),
  
  if Period > ?Treshold ->
%    io:format("~s Timeout, ask for work. ~n", [datetime_string()]),
    {F, BD, SD} = ask_for_work(),
    if F =:= Bhash ->
	       run_miners(Ports, Bhash, Period + ?TracePeriod);
	     true ->
         flush(),
         [Port ! {self(), {command, <<"S">>}} || {_Core_id, Port} <- Ports],
%         io:format("~s Sent command 'S'~n", [datetime_string()]),
         receive
           {_, {data, <<"S">>}} ->
             run_miners(Ports, F, BD, SD, 0);
           Err1 -> 
             io:format("S) Err from port: ~p", [Err1])
         after 10000 ->
           io:format("~s S) PORT timeout error: ~n", [datetime_string()])
         end
    end;
    true ->
      run_miners(Ports, Bhash, Period + ?TracePeriod)
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
%%      io:format("~s Server responce: ~256p~n", [datetime_string(), R]),
      R;
    _E -> 
      io:fwrite("\nIf you are running a solo-mining node, then this error may have happened because you need to turn on and sync your Amoveo node before you can mine. You can get it here: https://github.com/zack-bitcoin/amoveo \n If this error happens while connected to the public mining node, then it can probably be safely ignored."),
      timer:sleep(?Pool_sleep_period),
      talk_helper(Data, Peer, N-1)
  end.

check_data(Bhash, Nonce) ->
  io:format(">>> check data ~n", []),
  io:format("bhash: ~256p~n", [Bhash]),
  io:format("nonce: ~256p~n", [Nonce]),
  Y = <<Bhash/binary, Nonce/binary>>,
  io:format("Y: ~256p~n", [Y]),
  H = hash:doit(Y),
  I = pow:hash2integer(H, 1),
  io:format("~s check data: ~256p  ~p~n", [datetime_string(), H, I]).

  
datetime_string() ->
  {{_Year, Month, Day}, {Hour, Minute, Second}} = calendar:local_time(),
  lists:concat([Month, "/", Day, "-", Hour, ":", Minute, ":", Second]).
