%% @author axk456
%% @doc @todo Add description to miner_serv.

-module(miner_serv).
-include("miner.hrl").

-behaviour(gen_server).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2, code_change/3]).

%% ====================================================================
%% API functions
%% ====================================================================
-export([start/0]).

start() ->
  lager:start(),
  lager:info("  Let start mining.", []),
  os:cmd("pkill " ++ atom_to_list(?PORT_NAME)),
  timer:sleep(1000),
  gen_server:start({local, miner}, ?MODULE, [], []).

%% ====================================================================
%% Behavioural functions
%% ====================================================================

%% init/1
%% ====================================================================
%% @doc <a href="http://www.erlang.org/doc/man/gen_server.html#Module:init-1">gen_server:init/1</a>
-spec init(Args :: term()) -> Result when
	Result :: {ok, State}
			| {ok, State, Timeout}
			| {ok, State, hibernate}
			| {stop, Reason :: term()}
			| ignore,
	State :: term(),
	Timeout :: non_neg_integer() | infinity.
%% ====================================================================
init([]) ->
  Ports = [start_port(I) || I <- lists:seq(0, ?CORES - 1)],
  timer:apply_after(10, gen_server, cast, [miner, first_step]),
  {ok, #state{ports = Ports}}.

start_port(N) -> 
  Opts = [{packet, 4}, binary, exit_status, use_stdio, {args, [integer_to_list(N)]}],
  Port = erlang:open_port({spawn_executable, ?PORT_NAME}, Opts),
  {Port, N}.

%% handle_call/3
%% ====================================================================
%% @doc <a href="http://www.erlang.org/doc/man/gen_server.html#Module:handle_call-3">gen_server:handle_call/3</a>
-spec handle_call(Request :: term(), From :: {pid(), Tag :: term()}, State :: term()) -> Result when
	Result :: {reply, Reply, NewState}
			| {reply, Reply, NewState, Timeout}
			| {reply, Reply, NewState, hibernate}
			| {noreply, NewState}
			| {noreply, NewState, Timeout}
			| {noreply, NewState, hibernate}
			| {stop, Reason, Reply, NewState}
			| {stop, Reason, NewState},
	Reply :: term(),
	NewState :: term(),
	Timeout :: non_neg_integer() | infinity,
	Reason :: term().
%% ====================================================================

handle_call(_Request, _From, State) ->
  Reply = ok,
  {reply, Reply, State}.

%% handle_cast/2
%% ====================================================================
%% @doc <a href="http://www.erlang.org/doc/man/gen_server.html#Module:handle_cast-2">gen_server:handle_cast/2</a>
-spec handle_cast(Request :: term(), State :: term()) -> Result when
	Result :: {noreply, NewState}
			| {noreply, NewState, Timeout}
			| {noreply, NewState, hibernate}
			| {stop, Reason :: term(), NewState},
	NewState :: term(),
	Timeout :: non_neg_integer() | infinity.
%% ====================================================================
handle_cast(first_step, State) ->
  R1 = application:unload(miner_conf),
  lager:debug("After app unload. R1=~p~n", [R1]),
  R2 = application:load(miner_conf),
  lager:debug("After app load.   R2=~p~n", [R2]),
  GDIM = application:get_env(miner_conf, gdim, 32),
  BDIM = application:get_env(miner_conf, bdim, 96),
  lager:info("  GDIM= ~p,  BDIM= ~p.~n", [GDIM, BDIM]),

  {F, BD, _SD} = ask_for_work(),
  lager:debug(" ASK for work: Hash:~256p Diff:~p / ~p~n", [F, BD, _SD]),
  RS = crypto:strong_rand_bytes(23),
  [Port ! {self(), {command, <<"I", F/binary, RS/binary, BD:32/integer, GDIM:32/integer, BDIM:32/integer>>}} || {Port, _} <- State#state.ports],
  {noreply, State#state{command = <<"I">>, given_hash = F, difficulty = BD, msg_count = (?CORES - 1), period = ?Treshold}};

handle_cast(new_step, #state{given_hash = F, difficulty = BD} = State) ->
  R1 = application:unload(miner_conf),
  lager:debug("After app unload. R1=~p~n", [R1]),
  R2 = application:load(miner_conf),
  lager:debug("After app load.   R2=~p~n", [R2]),
  GDIM = application:get_env(miner_conf, gdim, 32),
  BDIM = application:get_env(miner_conf, bdim, 96),
  lager:info("  GDIM= ~p,  BDIM= ~p.~n", [GDIM, BDIM]),

  RS = crypto:strong_rand_bytes(23),
  [Port ! {self(), {command, <<"I", F/binary, RS/binary, BD:32/integer, GDIM:32/integer, BDIM:32/integer>>}} || {Port, _} <- State#state.ports],
  {noreply, State#state{command = <<"I">>, msg_count = (?CORES - 1)}};

handle_cast(_Msg, State) ->
  {noreply, State}.

%% handle_info/2
%% ====================================================================
%% @doc <a href="http://www.erlang.org/doc/man/gen_server.html#Module:handle_info-2">gen_server:handle_info/2</a>
-spec handle_info(Info :: timeout | term(), State :: term()) -> Result when
	Result :: {noreply, NewState}
			| {noreply, NewState, Timeout}
			| {noreply, NewState, hibernate}
			| {stop, Reason :: term(), NewState},
	NewState :: term(),
	Timeout :: non_neg_integer() | infinity.
%% ====================================================================
handle_info(tick, State) ->
  lager:info("  tick arrives.~n", []),
  [Port ! {self(), {command, <<"U">>}} || {Port, _Core_id} <- State#state.ports],
  {noreply, State#state{command = <<"U">>, msg_count = (?CORES - 1)}};

handle_info({Port, {data, <<"I">>}}, #state{msg_count = 0} = State) ->
  lager:info("  All ports have responded to command ~p. Last port responded is [~p].", ["I", Port]),
  self() ! tick, 
  {noreply, State};

handle_info({Port, {data, <<"U">>}}, #state{msg_count = 0, period = Period} = State)  when Period < ?Treshold ->
  lager:info("  All ports have responded to command ~p. Last port responded is [~p]. Period = ~p", ["U", Port, Period]),
  timer:send_after(?TracePeriod_1 * 1000, self(), tick),
  {noreply, State#state{period = (Period + ?TracePeriod_1)}};

handle_info({Port, {data, <<"U">>}}, #state{msg_count = 0, period = Period} = State) ->
  lager:info("  All ports have responded to command ~p. Last port responded is [~p]. Period( ~p) >= Threshold( ~p).", ["U", Port, Period, ?Treshold]),
  {F, BD, SD} = ask_for_work(),
  lager:debug("#~2.2.0w ask for work: Hash:~256p Diff:~p / ~p~n", [Period, F, BD, SD]),
  if F =:= State#state.given_hash ->
    timer:send_after(?TracePeriod_2 * 1000, self(), tick),
    {noreply, State#state{period = (Period + ?TracePeriod_2)}};
  true ->
    [Port ! {self(), {command, <<"S">>}} || {Port, _Core_id} <- State#state.ports],
    {noreply, State#state{command = <<"S">>, given_hash = F, difficulty = BD, msg_count = (?CORES - 1)}}
  end;

handle_info({Port, {data, <<"S">>}}, #state{msg_count = 0, period = Period} = State) ->
  lager:info("  All ports have responded to command ~p. Last port responded is [~p].", ["S", Port]),
  if Period =:= ?Treshold ->
    gen_server:cast(self(), new_step),
    {noreply, State#state{period = ?Treshold_1}};
  true ->
    gen_server:cast(self(), new_step),
    {noreply, State#state{period = 0}}
  end;

handle_info({Port, {data, <<1:8, Nonce:23/binary, _Hash:32/binary>>}}, State) ->
  lager:info("  Success data response from port[~p].", [Port]),
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
  PortIdx = proplists:get_value(Port, State#state.ports, -1),
  lager:warning("Port[~p] founds a nonce ~128p. Response from server: ~128p (diff=~p)~n", [PortIdx, BinNonce, LL, State#state.difficulty]),
  check_data(State#state.given_hash, Nonce),
  if State#state.msg_count == 0 ->
    timer:send_after(?TracePeriod_1 * 1000, self(), tick),
    {noreply, State#state{msg_count = 0}};
  true ->
    {noreply, State#state{msg_count = (State#state.msg_count - 1)}}
  end;

handle_info({Port, {data, Command}}, State) ->
  lager:info("  Port have responsed to command ~p. The response comes from port[~p].", [Command, Port]),
  {noreply, State#state{msg_count = (State#state.msg_count - 1)}};

handle_info(_Info, State) ->
  {noreply, State}.

%% terminate/2
%% ====================================================================
%% @doc <a href="http://www.erlang.org/doc/man/gen_server.html#Module:terminate-2">gen_server:terminate/2</a>
-spec terminate(Reason, State :: term()) -> Any :: term() when
	Reason :: normal
			| shutdown
			| {shutdown, term()}
			| term().
%% ====================================================================
terminate(_Reason, _State) ->
  ok.

%% code_change/3
%% ====================================================================
%% @doc <a href="http://www.erlang.org/doc/man/gen_server.html#Module:code_change-3">gen_server:code_change/3</a>
-spec code_change(OldVsn, State :: term(), Extra :: term()) -> Result when
	Result :: {ok, NewState :: term()} | {error, Reason :: term()},
	OldVsn :: Vsn | {down, Vsn},
	Vsn :: term().
%% ====================================================================
code_change(_OldVsn, State, _Extra) ->
  {ok, State}.

%% ====================================================================
%% Internal functions
%% ====================================================================

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

