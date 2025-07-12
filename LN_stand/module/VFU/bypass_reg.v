module bypass_reg #(
    parameter N = 4,
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [N*WIDTH-1:0] din_vec,
    output wire [N*WIDTH-1:0] dout_vec
);

genvar i;
generate
    for (i = 0; i < N; i = i + 1) begin : reg_array
        reg_1cycle #(.WIDTH(WIDTH)) u_reg (
            .clk(clk),
            .rst(rst),
            .en(en),
            .din(din_vec[i*WIDTH +: WIDTH]),
            .dout(dout_vec[i*WIDTH +: WIDTH])
        );
    end
endgenerate

endmodule
