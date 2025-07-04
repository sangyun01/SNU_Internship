module top #(parameter N = 4)(
    input  [N-1:0] x,
    input  [N-1:0] y,
    input          cin,
    output [N-1:0] sum,
    output         cout,
    output [2*N-1:0] prod
);

    cla_adder #(N) u_adder (
        .x(x),
        .y(y),
        .cin(cin),
        .sum(sum),
        .cout(cout)
    );

    unsigned_multiplier_gen #(N) u_mult (
        .x(x),
        .y(y),
        .prod(prod)
    );

endmodule
