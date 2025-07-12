module reg_mux #(
    parameter WIDTH = 16,
    parameter N = 4
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [1:0] inst,
    input  wire [N*WIDTH-1:0] add_result,
    input  wire [N*WIDTH-1:0] sub_result,
    input  wire [N*WIDTH-1:0] mult_result,
    input  wire [N*WIDTH-1:0] bypass_result,
    output reg  [N*WIDTH-1:0] out_result
);

wire [N*WIDTH-1:0] selected;
assign selected = (inst == 2'b00) ? mult_result  :
                  (inst == 2'b01) ? add_result    :
                  (inst == 2'b10) ? sub_result    :
                                   bypass_result;

always @(posedge clk) begin
    if (rst)
        out_result <= 0;
    else if (en)
        out_result <= selected;
end

endmodule

