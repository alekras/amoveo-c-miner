-module(miner)
.
-export([start/0, unpack_mining_data/1, speed_test/0]).
%-define(Peer, "http://localhost:3011/").%for a full node on same computer.
-define(Peer, "http://localhost:8081/").%for a full node on same computer.
%-define(Peer, "http://amoveopool2.com/work").%for a mining pool on the server.
%-define(Peer, "http://localhost:8085/").%for a mining pool on the same computer.
%-define(Peer, "http://159.89.106.253:8085/").%for a mining pool on the server.
-define(CORES, 2).
-define(Pubkey, <<"BAAfm5pn6ILYdMCt0zF5pE2E82jsBV4ZJPbwkgM3DBS2I+/hiJc5yCnb9rrlfpWasDOB/oCYfRJ63CBG1GbAUqI=">>).
-define(period, 40000).%how long to wait in seconds before checking if new mining data is available.
-define(pool_sleep_period, 1000).%How long to wait in miliseconds if we cannot connect to the mining pool.
-define(miner_sleep, 0). %This is how you reduce the load on CPU. It sleeps this long in miliseconds between mining cycles.
%%-define(HTTPC, httpc_mock).
-define(HTTPC, httpc).
-define(PORT_NAME, amoveo_c_miner).
-define(USE_SHARE_POOL, false).

start() ->
  io:format("~p Started mining.~n", [datetime_string()]),
  os:cmd("pkill " ++ atom_to_list(?PORT_NAME)),
  start2().

start2() ->
  Data = case ?USE_SHARE_POOL of
    true -> <<"[\"mining_data\",\"", ?Pubkey/binary, "\"]">>;
    false -> <<"[\"mining_data\"]">>
	end,
  R = talk_helper(Data, ?Peer, 1000),
  io:format("R = ~128p~n", [R]),
  if is_list(R) ->
       start_c_miners(R);
     true ->
	     timer:sleep(1000),
	     start()
  end.

flush() ->
  receive
    _ -> flush()
  after 0 -> ok
  end.

unpack_mining_data(R) ->
  [<<"ok">>, [_, Hash, BlockDiff, ShareDiff]] = mochijson2:decode(R),
  case ?USE_SHARE_POOL of
	  true ->
      F = base64:decode(Hash),
      {F, BlockDiff, ShareDiff};
		false ->
      F = base64:decode(Hash),
      {F, ShareDiff, ShareDiff}
	end.

start_c_miners(R) ->
  {F, BD, SD} = unpack_mining_data(R), %S is the nonce
  RS = crypto:strong_rand_bytes(32),
  flush(),
  Ports = start_many(?CORES),
	[Port ! {self(), {command, <<F/binary, RS/binary, BD:32/integer, SD:32/integer, Core_id:32/integer>>}} || {Core_id, Port} <- Ports],
  receive
		{_Port, {data, Nonce}} ->
      io:format("Data from port: ~p", [Nonce]),
      case Nonce of 
        0 -> io:fwrite("nonce 0 error\n");
        _ ->
		      BinNonce = base64:encode(Nonce),
		      Data = <<"[\"work\",\"", BinNonce/binary, "\",\"", ?Pubkey/binary, "\"]">>,
          RR = talk_helper(Data, ?Peer, 5),
          LL = try mochijson2:decode(RR) of
            [_ | L] -> L;
			      L -> L
		      catch E:Reason ->
            io:format(" ---- Unexpected response from server: ~128p~nError: ~p : ~p~n", [RR, E, Reason]),
            RR
          end,
          io:format("~p Found a block. Nonce ~128p. Response from server: ~128p~n~n", [datetime_string(), BinNonce, LL]),
		      timer:sleep(200)
	    end;
    Err -> 
      io:format("Err from port: ~p", [Err])
  after (?period * 1000) ->
    io:fwrite("did not find a block in that period \n")
  end,
	[Port ! {self(), close} || {_, Port} <- Ports],
  timer:sleep(?miner_sleep),
  start2().

start_many(N) when N < 1 -> [];
start_many(N) -> 
  Opts = [{packet, 4}, binary, exit_status, use_stdio],
  Port = erlang:open_port({spawn_executable, ?PORT_NAME}, Opts),
  [{N, Port} | start_many(N - 1)].

talk_helper2(Data, Peer) ->
  ?HTTPC:request(post, {Peer, [], "application/octet-stream", iolist_to_binary(Data)}, [{timeout, 3000}], []).

talk_helper(_Data, _Peer, 0) -> throw("talk helper failed");
talk_helper(Data, Peer, N) ->
  case talk_helper2(Data, Peer) of
    {ok, {_Status, _Headers, []}} ->
      io:fwrite("server gave confusing response\n"),
      timer:sleep(?pool_sleep_period),
      talk_helper(Data, Peer, N-1);
    {ok, {_, _, R}} ->
      io:format("server responce: ~256p~n", [R]),
      R;
    _E -> 
      io:fwrite("\nIf you are running a solo-mining node, then this error may have happened because you need to turn on and sync your Amoveo node before you can mine. You can get it here: https://github.com/zack-bitcoin/amoveo \n If this error happens while connected to the public mining node, then it can probably be safely ignored."),
      timer:sleep(?pool_sleep_period),
      talk_helper(Data, Peer, N-1)
  end.

datetime_string() ->
  {{_Year, Month, Day}, {Hour, Minute, Second}} = calendar:local_time(),
  lists:concat([Month, "/", Day, "-", Hour, ":", Minute, ":", Second]).

speed_test() -> 
  io:format("~p Speed test started.~n", [datetime_string()]),
  BD = 10,
  SD = 10,
  F = <<0:256>>,
  RS = <<0:256>>,
  file:write_file("mining_input", <<F/binary, RS/binary, BD:32/integer, SD:32/integer>>).
