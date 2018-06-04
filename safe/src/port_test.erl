%% @author alexei
%% @doc @todo Add description to port_test.

-module(port_test).

%% ====================================================================
%% API functions
%% ====================================================================
-export([start/0, stop/0, init/1]).
-export([foo/1, bar/1]).

start() ->
    spawn(?MODULE, init, ["extprg"]).
stop() ->
    complex ! stop.

foo(X) ->
    call_port({foo, X}).
bar(Y) ->
    call_port({bar, Y}).

call_port(Msg) ->
  complex ! {call, self(), Msg},
  receive
    {complex, Result} ->
	    Result
  after 1000 -> io:format("no response. msg=~p~n", [Msg])
  end.

init(ExtPrg) ->
    register(complex, self()),
    process_flag(trap_exit, true),
    Port = open_port({spawn_executable, ExtPrg}, [{packet, 4}, binary, exit_status, use_stdio]),
		timer:sleep(100),
		io:format("start port: name=~p port=~p~n~n", [ExtPrg, Port]),
    loop(Port).

loop(Port) ->
    receive
	{call, Caller, Msg} ->
	    Port ! {self(), {command, encode(Msg)}},
	    receive
        {Port, {data, Data}} ->
          Caller ! {complex, decode(Data)};
				A -> io:format("response A from port. =~p~n", [A])
      after 500 -> io:format("no response from port. msg=~p~n", [Msg])
	    end,
	    loop(Port);
	stop ->
	    Port ! {self(), close},
	    receive
		{Port, closed} ->
		    exit(normal)
	    end;
	{'EXIT', Port, Reason} ->
	    exit(port_terminated)
    end.

encode({foo, X}) -> <<1:8, X:8, 0:8>>;
encode({bar, Y}) -> <<2:8, Y:8>>.

decode(M) -> M.


%% ====================================================================
%% Internal functions
%% ====================================================================


