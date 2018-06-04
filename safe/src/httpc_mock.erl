%% @author alexei
%% @doc @todo Add description to httpc_mock.


-module(httpc_mock).

%% ====================================================================
%% API functions
%% ====================================================================
-export([request/4]).

request(Method, Url, Opt1, Opt2) ->
  io:format("HTTPC_MOCK: request: ~p ~p ~p ~p~n", [Method, Url, Opt1, Opt2]),
  {ok, {[], [],
	"[\"ok\",[-6,\"7QL+S+jS4e+3T9dnuL1MDn2EtPK4Udg8R7RXOH3ji34=\",\"tOA6AeIYHShTwyIorv9UeHcdd32b3idM9DnhweK0N1Q=\",8844]]"
	 }}.

%% ====================================================================
%% Internal functions
%% ====================================================================


