%% @author alexei
%% @doc @todo Add description to httpc_mock.


-module(httpc_mock).

%% ====================================================================
%% API functions
%% ====================================================================
-export([request/4, loop/1]).

request(Method, Url, Opt1, Opt2) ->
%%  io:format("HTTPC_MOCK: request: ~p ~p ~p ~p~n", [Method, Url, Opt1, Opt2]),
  case whereis(http_mock) of
    Pid when is_pid(Pid) -> ok;
	  _ ->
		   init()
  end,
  http_mock ! {get, self()},
	receive
    {bh,BH} -> ok
  end,
%%  io:format("HTTPC_MOCK: response: ~p~n", [BH]),
  {ok, {[], [],
	"[\"ok\",[-6,\""++ BH ++ "\",8844,8844]]"
	 }}.

init() ->
  io:format("HTTPC_MOCK: init.~n", []),
  Pid = spawn(?MODULE, loop, [0]),
  register(http_mock, Pid),
  timer:send_interval((10 * 1000), http_mock, set).

%% ====================================================================
%% Internal functions
%% ====================================================================

loop(N) ->
	receive
	  {get, Pid} when N == 0 ->
			Pid ! {bh,"7QL+S+jS4e+3T9dnuL1MDn2EtPK4Udg8R7RXOH3ji34="},
			loop(N);
	  {get, Pid} when N == 1 ->
		  Pid ! {bh,"8WL+S+jS4e+3T9dnuL1MDn2EtPK4Udg8R7RXOH3ji34="},
		  loop(N);
	  set -> loop((N + 1) rem 2)
	end.