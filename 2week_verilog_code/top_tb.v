`timescale 1ns / 1ps

module top_tb;

    parameter N = 4;
    parameter M = 16;

    reg  [N-1:0] x, y;
    reg          cin;
    wire [N-1:0] sum;
    wire         cout;
    wire [2*N-1:0] prod;

    reg [N-1:0] x_array             [0:M-1];
    reg [N-1:0] y_array             [0:M-1];
    reg [N:0]   expected_sum_array  [0:M-1];  // {cout, sum}
    reg [2*N-1:0] expected_prod_array [0:M-1];

    integer i;

    // 상위 모듈 인스턴스
    top UUT(
        .x   (x),
        .y   (y),
        .cin (cin),
        .sum (sum),
        .cout(cout),
        .prod(prod)
    );

    initial begin
        $readmemh("in_x.data", x_array);
        $readmemh("in_y.data", y_array);
        $readmemh("out_sum.data", expected_sum_array);
        $readmemh("out_prod.data", expected_prod_array);
    end

    initial begin
        #20;
        for (i = 0; i <= M - 1; i = i + 1) begin
            x = x_array[i];
            y = y_array[i];
            cin = 1'b0;
            #15;
            if (expected_sum_array[i] != {cout, sum} || expected_prod_array[i] != prod)
                $display("Error iteration %h\n", i);
            #5;
        end
    end

    initial begin
        #200 $finish;
    end

    initial begin
        $monitor($realtime, "ns %d %d %d %d %d", x, y, cin, {cout, sum}, prod);
    end

endmodule
